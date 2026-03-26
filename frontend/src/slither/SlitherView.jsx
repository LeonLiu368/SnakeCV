import { useEffect, useRef, useState } from 'react'
import { PELLET_RADIUS, HEAD_RADIUS, BODY_RADIUS, MAGNET_RADIUS, toroidalDistSq } from './slitherLogic.js'
import shieldPowerupIcon from '../assets/powerups/shield.svg'
import ghostPowerupIcon from '../assets/powerups/ghost.svg'
import magnetPowerupIcon from '../assets/powerups/magnet.svg'

/** Parse color to [r,g,b]; accepts #hex or rgb(r,g,b). */
function parseColorToRgb(c) {
  if (c.startsWith('rgb')) {
    const match = c.match(/\d+/g)
    return match ? match.slice(0, 3).map(Number) : [255, 255, 255]
  }
  const n = parseInt(c.replace('#', ''), 16)
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

/** Mix color with white; amount 0 = original, 1 = white. Accepts #hex or rgb(). */
function brightenColor(hexOrRgb, amount) {
  const [r, g, b] = parseColorToRgb(hexOrRgb)
  const r2 = Math.round(r + (255 - r) * amount)
  const g2 = Math.round(g + (255 - g) * amount)
  const b2 = Math.round(b + (255 - b) * amount)
  return `rgb(${r2},${g2},${b2})`
}

const GHOST_TINT = '#c4a8f0'
/** Blend hex color with ghost tint; factor 0 = original, 1 = ghost. */
function blendWithGhost(hex, factor = 0.55) {
  const n = parseInt(hex.replace('#', ''), 16)
  const m = parseInt(GHOST_TINT.replace('#', ''), 16)
  const r = Math.round(((n >> 16) & 0xff) + (((m >> 16) & 0xff) - ((n >> 16) & 0xff)) * factor)
  const g = Math.round(((n >> 8) & 0xff) + (((m >> 8) & 0xff) - ((n >> 8) & 0xff)) * factor)
  const b_ = Math.round((n & 0xff) + ((m & 0xff) - (n & 0xff)) * factor)
  return `rgb(${r},${g},${b_})`
}

const EPS = 1e-6
/** Allow crossings at segment endpoints (t === 0 or t === 1) so we split at the boundary until the last piece has crossed. */
const T_MIN = -1e-6
const T_MAX = 1 + 1e-6

const ORB_GLOW_BLUR = 12
const ORB_HIGHLIGHT_OFFSET = 0.35

/** Power-up pellet appearance: shadow color, fill color for map icons, and gradient stops. */
const POWERUP_STYLE = {
  shield: {
    shadowColor: 'rgba(100, 180, 255, 0.95)',
    iconColor: '#5090ff',
    gradient: [
      [0, 'rgba(255, 255, 255, 1)'],
      [0.25, 'rgba(180, 220, 255, 1)'],
      [0.6, 'rgba(80, 160, 255, 1)'],
      [1, 'rgba(40, 100, 200, 1)'],
    ],
  },
  ghost: {
    shadowColor: 'rgba(200, 180, 255, 0.95)',
    iconColor: '#a078dc',
    gradient: [
      [0, 'rgba(255, 255, 255, 1)'],
      [0.3, 'rgba(230, 210, 255, 1)'],
      [0.7, 'rgba(160, 120, 220, 1)'],
      [1, 'rgba(100, 60, 180, 1)'],
    ],
  },
  magnet: {
    shadowColor: 'rgba(255, 200, 80, 0.95)',
    iconColor: '#e8b030',
    gradient: [
      [0, 'rgba(255, 255, 255, 1)'],
      [0.2, 'rgba(255, 240, 180, 1)'],
      [0.5, 'rgba(255, 200, 60, 1)'],
      [1, 'rgba(220, 140, 20, 1)'],
    ],
  },
}
const ORB_SPECULAR_OFFSET = 0.15
const ORB_SPECULAR_RADIUS = 0.28
const HEAD_EYE_OFFSET = HEAD_RADIUS * 0.17
const HEAD_SPECULAR_OFFSET = HEAD_RADIUS * 0.17
const HEAD_SPECULAR_RADIUS = HEAD_RADIUS * 0.2

/**
 * Return 1 or 2 line segments to draw a toroidal line from p1 to p2.
 * Draw as two segments when the segment crosses a boundary; as one only when both
 * endpoints are on the same side (i.e. the last piece has crossed the boundary).
 * Each segment is [x1, y1, x2, y2] in world coords within bounds.
 * @param {{ x: number, y: number }} p1
 * @param {{ x: number, y: number }} p2
 * @param {{ width: number, height: number }} bounds
 * @returns {Array<[number, number, number, number]>}
 */
function toroidalLineSegments(p1, p2, bounds) {
  const w = bounds.width
  const h = bounds.height
  const rawDx = p2.x - p1.x
  const rawDy = p2.y - p1.y
  let dx = rawDx - w * Math.round(rawDx / w)
  let dy = rawDy - h * Math.round(rawDy / h)
  const crosses = []
  if (Math.abs(dx) > EPS) {
    const t0 = (0 - p1.x) / dx
    if (t0 >= T_MIN && t0 <= T_MAX) crosses.push({ t: t0, x: 0, y: p1.y + t0 * dy, edge: 'x0' })
    const t1 = (w - p1.x) / dx
    if (t1 >= T_MIN && t1 <= T_MAX) crosses.push({ t: t1, x: w, y: p1.y + t1 * dy, edge: 'x1' })
  }
  if (Math.abs(dy) > EPS) {
    const t0 = (0 - p1.y) / dy
    if (t0 >= T_MIN && t0 <= T_MAX) crosses.push({ t: t0, x: p1.x + t0 * dx, y: 0, edge: 'y0' })
    const t1 = (h - p1.y) / dy
    if (t1 >= T_MIN && t1 <= T_MAX) crosses.push({ t: t1, x: p1.x + t1 * dx, y: h, edge: 'y1' })
  }
  // Segment straddles boundary but |dx| or |dy| is below EPS (both points very close to opposite edges).
  // Force split so we draw two segments until fully crossed.
  const straddleX = Math.abs(rawDx) > w / 2
  const straddleY = Math.abs(rawDy) > h / 2
  if (crosses.length === 0) {
    if (straddleX && !straddleY) {
      return [
        [p1.x, p1.y, w, p1.y],
        [0, p1.y, p2.x, p2.y],
      ]
    }
    if (straddleY && !straddleX) {
      return [
        [p1.x, p1.y, p1.x, h],
        [p1.x, 0, p2.x, p2.y],
      ]
    }
    if (straddleX && straddleY) {
      // Corner: split on both; use x first then y for consistent ordering.
      return [
        [p1.x, p1.y, w, p1.y],
        [0, p1.y, p2.x, p2.y],
      ]
    }
    return [[p1.x, p1.y, p2.x, p2.y]]
  }
  crosses.sort((a, b) => a.t - b.t)
  const texit = crosses[0]
  const tenter = crosses.length >= 2 ? crosses[crosses.length - 1] : null
  const exitX = texit.x
  const exitY = texit.y
  let reX = tenter ? tenter.x : exitX
  let reY = tenter ? tenter.y : exitY
  if (tenter) {
    if (tenter.edge === 'x0') reX = w
    else if (tenter.edge === 'x1') reX = 0
    else if (tenter.edge === 'y0') reY = h
    else if (tenter.edge === 'y1') reY = 0
  } else {
    if (texit.edge === 'x0') reX = w
    else if (texit.edge === 'x1') reX = 0
    else if (texit.edge === 'y0') reY = h
    else if (texit.edge === 'y1') reY = 0
  }
  return [
    [p1.x, p1.y, exitX, exitY],
    [reX, reY, p2.x, p2.y],
  ]
}

/** Draw a toroidal polyline (snake body) with ctx.moveTo/lineTo. */
function strokeToroidalPath(ctx, points, bounds) {
  if (points.length < 2) return
  for (let i = 1; i < points.length; i++) {
    const parts = toroidalLineSegments(points[i - 1], points[i], bounds)
    for (const s of parts) {
      ctx.moveTo(s[0], s[1])
      ctx.lineTo(s[2], s[3])
    }
  }
}

/** Length of a segment [x1, y1, x2, y2]. */
function segmentLength(s) {
  const dx = s[2] - s[0]
  const dy = s[3] - s[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Draw toroidal path with a gradient that follows the path (head to tail).
 * Keeps gradient clean when the snake crosses map boundaries.
 */
function strokeToroidalPathWithGradient(ctx, points, bounds, colorHead, colorTail, lineWidth) {
  if (points.length < 2) return
  const parts = []
  for (let i = 1; i < points.length; i++) {
    const segs = toroidalLineSegments(points[i - 1], points[i], bounds)
    for (const s of segs) parts.push(s)
  }
  let totalLen = 0
  for (const s of parts) totalLen += segmentLength(s)
  if (totalLen <= 0) return
  const parseColor = (c) => {
    if (c.startsWith('rgb')) {
      const match = c.match(/\d+/g)
      return match ? match.slice(0, 3).map(Number) : [255, 255, 255]
    }
    const n = parseInt(c.replace('#', ''), 16)
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
  }
  const lerp = (a, b, t) => {
    const [r1, g1, b1] = parseColor(a)
    const [r2, g2, b2] = parseColor(b)
    return `rgb(${Math.round(r1 + (r2 - r1) * t)},${Math.round(g1 + (g2 - g1) * t)},${Math.round(b1 + (b2 - b1) * t)})`
  }
  let run = 0
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.lineWidth = lineWidth
  for (const s of parts) {
    const len = segmentLength(s)
    const t0 = run / totalLen
    const t1 = (run + len) / totalLen
    run += len
    const grad = ctx.createLinearGradient(s[0], s[1], s[2], s[3])
    grad.addColorStop(0, lerp(colorHead, colorTail, t0))
    grad.addColorStop(1, lerp(colorHead, colorTail, t1))
    ctx.strokeStyle = grad
    ctx.beginPath()
    ctx.moveTo(s[0], s[1])
    ctx.lineTo(s[2], s[3])
    ctx.stroke()
  }
}

/** Margin from edge (world units) at which to also draw on the opposite side. */
const WRAP_DRAW_MARGIN = 120

/**
 * Yield (x, y) positions at which to draw a point so it appears on both sides when near an edge.
 * First yield is always (px, py); then wrapped positions for edges the point is near.
 */
function* toroidalDrawPositions(px, py, bounds) {
  yield [px, py]
  const w = bounds.width
  const h = bounds.height
  if (px < WRAP_DRAW_MARGIN) yield [px + w, py]
  if (px > w - WRAP_DRAW_MARGIN) yield [px - w, py]
  if (py < WRAP_DRAW_MARGIN) yield [px, py + h]
  if (py > h - WRAP_DRAW_MARGIN) yield [px, py - h]
}

/**
 * Canvas renderer for slither game state.
 * Camera follows the longest snake or the player snake; reports mouse position in world coords.
 * @param {{ state: { snakes: Array<{ segments: Array<{x,y}>, color: string, isPlayer?: boolean }>, pellets: Array<{x,y}>, bounds: { width, height } } }, onMouseMove?: (worldX: number, worldY: number) => void, playerDeadSnake?: { segments: Array<{x,y}>, color: string } | null, deathAnimationProgress?: number | null, speedBoostActive?: boolean, speedBoostProgress?: number }} props
 */
const MINIMAP_SIZE = 140
const MINIMAP_MARGIN = 4

/** Ease-out cubic for snappy start and smooth end. */
function easeOutCubic(t) {
  return 1 - (1 - t) ** 3
}

/** Draw a snake with death animation (eased shrink + fade + head burst). */
function drawDeathAnimation(ctx, snake, progress, bounds, scale, offset) {
  const segs = snake.segments
  if (segs.length < 2) return
  const eased = easeOutCubic(progress)
  const head = segs[0]
  const s = Math.max(0.01, 1 - 0.5 * eased)
  ctx.save()
  ctx.globalAlpha = 1 - eased
  ctx.translate(head.x, head.y)
  ctx.scale(s, s)
  ctx.translate(-head.x, -head.y)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const bodyWidth = (BODY_RADIUS * 2) / scale
  ctx.beginPath()
  strokeToroidalPath(ctx, segs, bounds)
  ctx.strokeStyle = 'rgba(0,0,0,0.2)'
  ctx.lineWidth = bodyWidth + 6 / scale
  ctx.stroke()
  strokeToroidalPathWithGradient(
    ctx,
    segs,
    bounds,
    brightenColor(snake.color, 0.35),
    snake.color,
    bodyWidth,
  )
  ctx.fillStyle = snake.color
  const headPositions =
    offset.x === 0 && offset.y === 0
      ? toroidalDrawPositions(head.x, head.y, bounds)
      : [[head.x, head.y]]
  for (const [hx, hy] of headPositions) {
    ctx.beginPath()
    ctx.arc(hx, hy, HEAD_RADIUS, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = 'rgba(255,255,255,0.45)'
    ctx.lineWidth = 1.2 / scale
    ctx.stroke()
    ctx.fillStyle = snake.color
  }
  ctx.restore()

  if (progress < 0.4) {
    const burstAlpha = 1 - progress / 0.4
    const burstRadius = HEAD_RADIUS * (1 + progress * 3)
    const burstColor = brightenColor(snake.color, 0.6)
    ctx.save()
    ctx.globalAlpha = burstAlpha
    ctx.strokeStyle = burstColor
    ctx.lineWidth = (4 / scale)
    for (const [hx, hy] of headPositions) {
      ctx.beginPath()
      ctx.arc(hx, hy, burstRadius, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.restore()
  }
}

/** Unwrap focus so camera moves continuously when head crosses boundary (no jump). */
function unwrapFocus(focus, prev, w, h) {
  if (prev == null || prev.x == null) return { x: focus.x, y: focus.y }
  let x = focus.x
  let y = focus.y
  const dx = focus.x - prev.x
  const dy = focus.y - prev.y
  if (dx > w / 2) x = focus.x - w
  else if (dx < -w / 2) x = focus.x + w
  if (dy > h / 2) y = focus.y - h
  else if (dy < -h / 2) y = focus.y + h
  return { x, y }
}

const POWERUP_ICON_URLS = { shield: shieldPowerupIcon, ghost: ghostPowerupIcon, magnet: magnetPowerupIcon }

export function SlitherView({
  state,
  onMouseMove,
  onCanvasBoost,
  playerDeadSnake,
  deathAnimationProgress,
  botDeadSnakes = [],
  deathAnimMs = 1200,
  speedBoostActive,
  speedBoostProgress,
  cameraShakeX = 0,
  cameraShakeY = 0,
}) {
  const canvasRef = useRef(null)
  const minimapRef = useRef(null)
  const wrapRef = useRef(null)
  const cameraRef = useRef({ scale: 1, camX: 0, camY: 0 })
  const cameraWorldRef = useRef(null)
  const [resizeTick, setResizeTick] = useState(0)
  const powerupImagesRef = useRef({ shield: null, ghost: null, magnet: null })
  const [powerupImagesReady, setPowerupImagesReady] = useState(false)

  useEffect(() => {
    const urls = POWERUP_ICON_URLS
    const keys = Object.keys(urls)
    let loaded = 0
    const images = { shield: null, ghost: null, magnet: null }
    keys.forEach((key) => {
      const img = new window.Image()
      img.onload = () => {
        images[key] = img
        loaded++
        if (loaded === keys.length) {
          powerupImagesRef.current = images
          setPowerupImagesReady(true)
        }
      }
      img.src = urls[key]
    })
  }, [])

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return
    const setSize = () => {
      const w = Math.max(1, wrap.clientWidth)
      const h = Math.max(1, wrap.clientHeight)
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
        setResizeTick((t) => t + 1)
      }
    }
    setSize()
    const raf1 = requestAnimationFrame(() => {
      setSize()
      requestAnimationFrame(setSize)
    })
    const ResizeObserverCtor = typeof window !== 'undefined' ? window.ResizeObserver : null
    if (!ResizeObserverCtor) return () => cancelAnimationFrame(raf1)
    const ro = new ResizeObserverCtor(setSize)
    ro.observe(wrap)
    return () => {
      cancelAnimationFrame(raf1)
      ro.disconnect()
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !state) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!state.bounds || state.bounds.width <= 0 || state.bounds.height <= 0) return
    if (canvas.width === 0 || canvas.height === 0) {
      const wrap = wrapRef.current
      if (wrap && wrap.clientWidth > 0 && wrap.clientHeight > 0) {
        canvas.width = Math.max(1, wrap.clientWidth)
        canvas.height = Math.max(1, wrap.clientHeight)
        requestAnimationFrame(() => setResizeTick((t) => t + 1))
      }
      return
    }

    const { snakes, pellets, bounds } = state

    const player = snakes.find((s) => s.isPlayer)
    const longest = snakes.reduce(
      (best, s) => (s.segments.length > (best?.segments.length ?? 0) ? s : best),
      null,
    )
    const focus =
      (playerDeadSnake?.segments?.[0] ??
        (player ?? longest)?.segments?.[0]) ??
      { x: bounds.width / 2, y: bounds.height / 2 }

    const w = bounds.width
    const h = bounds.height
    const cameraWorld = unwrapFocus(focus, cameraWorldRef.current, w, h)
    cameraWorldRef.current = cameraWorld

    const VIEW_WORLD_SIZE = 5000
    let scale = Math.min(
      canvas.width / VIEW_WORLD_SIZE,
      canvas.height / VIEW_WORLD_SIZE,
      1.8,
    )
    if (speedBoostActive && speedBoostProgress != null) {
      scale = scale * (1 - 0.15 * Math.sin(speedBoostProgress * Math.PI))
    }
    const camX = canvas.width / 2 - cameraWorld.x * scale
    const camY = canvas.height / 2 - cameraWorld.y * scale
    cameraRef.current = { scale, camX, camY }

    ctx.save()
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.fillStyle = '#0d0d0d'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.translate(cameraShakeX, cameraShakeY)
    ctx.translate(camX, camY)
    ctx.scale(scale, scale)

    const tNow = performance.now() / 1000
    const pelletPulse = 1 + 0.08 * Math.sin(tNow * Math.PI)
    const goldenPulse = 1 + 0.12 * Math.sin(tNow * Math.PI * 2.2)
    const toroidalOffsets = [
      { x: 0, y: 0 },
      { x: -w, y: 0 },
      { x: w, y: 0 },
      { x: 0, y: -h },
      { x: 0, y: h },
      { x: -w, y: -h },
      { x: -w, y: h },
      { x: w, y: -h },
      { x: w, y: h },
    ]

    for (const offset of toroidalOffsets) {
      ctx.save()
      ctx.translate(offset.x, offset.y)

      const collectRadius = HEAD_RADIUS + PELLET_RADIUS + MAGNET_RADIUS
      const magnetPull = 0.32
      const magnetShrink = 0.38
      const magnetFade = 0.28
      for (const pellet of pellets) {
        let nearestHead = focus
        let distSq = toroidalDistSq(focus, pellet, bounds)
        for (const snake of snakes) {
          const head = snake.segments[0]
          if (!head) continue
          const d2 = toroidalDistSq(head, pellet, bounds)
          if (d2 < distSq) {
            distSq = d2
            nearestHead = head
          }
        }
        const dist = Math.sqrt(distSq)
        const t = Math.max(0, 1 - dist / collectRadius)
        const absorption = t * t * (3 - 2 * t)
        const pullT = Math.pow(t, 0.85)
        const pullCurve = pullT * pullT * (2 - pullT)
        let pullX = 0
        let pullY = 0
        if (pullCurve > 0) {
          pullX = (nearestHead.x - pellet.x) - w * Math.round((nearestHead.x - pellet.x) / w)
          pullY = (nearestHead.y - pellet.y) - h * Math.round((nearestHead.y - pellet.y) / h)
          const pull = pullCurve * magnetPull
          pullX *= pull
          pullY *= pull
        }
        const valueMult = 1 + (pellet.value - 1) * 0.2
        const sizeMult = 1 - absorption * magnetShrink
        const alpha = 1 - absorption * magnetFade
        const positions =
          offset.x === 0 && offset.y === 0
            ? toroidalDrawPositions(pellet.x, pellet.y, bounds)
            : [[pellet.x, pellet.y]]
        const ptype = pellet.type ?? 'normal'
        const isGolden = ptype === 'golden'
        const isTrail = ptype === 'trail'
        const isPowerUp = ptype === 'shield' || ptype === 'ghost' || ptype === 'magnet'
        const powerUpStyle = isPowerUp ? POWERUP_STYLE[ptype] : null
        const sizeScale = isPowerUp ? 1.25 : isGolden ? 1.5 : isTrail ? 0.72 : 1
        const usePull = offset.x === 0 && offset.y === 0
        const powerupImgs = powerupImagesRef.current
        const powerupImg = isPowerUp ? powerupImgs[ptype] : null
        const usePowerupIcon = isPowerUp && powerupImg && powerupImagesReady
        const pulseForPellet = isGolden ? goldenPulse : pelletPulse
        for (const [px, py] of positions) {
          const drawX = usePull ? px + pullX : px
          const drawY = usePull ? py + pullY : py
          const r = PELLET_RADIUS * sizeScale * pulseForPellet * sizeMult
          ctx.save()
          ctx.globalAlpha = alpha
          if (isGolden) {
            const highlightOffset = r * ORB_HIGHLIGHT_OFFSET
            const cx = drawX - highlightOffset
            const cy = drawY - highlightOffset
            ctx.shadowColor = 'rgba(255, 215, 80, 0.95)'
            ctx.shadowBlur = (ORB_GLOW_BLUR * 1.8 * sizeMult) / scale
            const orbGrad = ctx.createRadialGradient(cx, cy, 0, drawX, drawY, r)
            orbGrad.addColorStop(0, 'rgba(255, 255, 255, 1)')
            orbGrad.addColorStop(0.2, 'rgba(255, 248, 200, 1)')
            orbGrad.addColorStop(0.5, 'rgba(255, 200, 60, 1)')
            orbGrad.addColorStop(0.85, 'rgba(220, 140, 20, 1)')
            orbGrad.addColorStop(1, 'rgba(180, 90, 30, 1)')
            ctx.fillStyle = orbGrad
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.fill()
            ctx.strokeStyle = 'rgba(255, 255, 220, 0.85)'
            ctx.lineWidth = 2.2 / scale
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.stroke()
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
            ctx.beginPath()
            ctx.arc(cx - r * 0.12, cy - r * 0.12, r * 0.22, 0, Math.PI * 2)
            ctx.fill()
            ctx.shadowBlur = 0
          } else if (isTrail) {
            const highlightOffset = r * ORB_HIGHLIGHT_OFFSET
            const cx = drawX - highlightOffset
            const cy = drawY - highlightOffset
            ctx.shadowColor = 'rgba(80, 200, 255, 0.5)'
            ctx.shadowBlur = (ORB_GLOW_BLUR * 0.45 * sizeMult) / scale
            const orbGrad = ctx.createRadialGradient(cx, cy, 0, drawX, drawY, r)
            orbGrad.addColorStop(0, 'rgba(200, 245, 255, 1)')
            orbGrad.addColorStop(0.35, 'rgba(100, 200, 255, 0.95)')
            orbGrad.addColorStop(0.75, 'rgba(40, 140, 220, 0.9)')
            orbGrad.addColorStop(1, 'rgba(20, 80, 160, 0.85)')
            ctx.fillStyle = orbGrad
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.fill()
            ctx.strokeStyle = 'rgba(180, 230, 255, 0.45)'
            ctx.lineWidth = 0.8 / scale
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.stroke()
            ctx.shadowBlur = 0
          } else if (usePowerupIcon) {
            const iconSize = PELLET_RADIUS * 3.8 * sizeMult
            const half = iconSize / 2
            const x = drawX - half
            const y = drawY - half
            const baseR = half + 8 / scale
            if (ptype === 'shield') {
              ctx.fillStyle = powerUpStyle.iconColor + '40'
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR, 0, Math.PI * 2)
              ctx.fill()
              ctx.strokeStyle = powerUpStyle.iconColor
              ctx.lineWidth = 4 / scale
              ctx.stroke()
              ctx.shadowColor = powerUpStyle.shadowColor
              ctx.shadowBlur = (14 * sizeMult) / scale
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR - 2 / scale, 0, Math.PI * 2)
              ctx.stroke()
            } else if (ptype === 'ghost') {
              ctx.shadowColor = powerUpStyle.shadowColor
              ctx.shadowBlur = (22 * sizeMult) / scale
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR + 4 / scale, 0, Math.PI * 2)
              ctx.fillStyle = powerUpStyle.iconColor + '25'
              ctx.fill()
              ctx.shadowBlur = (12 * sizeMult) / scale
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR, 0, Math.PI * 2)
              ctx.fill()
              ctx.strokeStyle = 'rgba(255,255,255,0.35)'
              ctx.lineWidth = 1.5 / scale
              ctx.setLineDash([6 / scale, 4 / scale])
              ctx.stroke()
              ctx.setLineDash([])
            } else {
              ctx.strokeStyle = powerUpStyle.iconColor + 'cc'
              ctx.lineWidth = 3 / scale
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR + 3 / scale, 0, Math.PI * 2)
              ctx.stroke()
              ctx.strokeStyle = 'rgba(255,255,255,0.5)'
              ctx.lineWidth = 2 / scale
              ctx.beginPath()
              ctx.arc(drawX, drawY, baseR - 2 / scale, 0, Math.PI * 2)
              ctx.stroke()
              ctx.shadowColor = powerUpStyle.shadowColor
              ctx.shadowBlur = (10 * sizeMult) / scale
            }
            ctx.shadowColor = 'transparent'
            ctx.shadowBlur = 0
            ctx.drawImage(powerupImg, x, y, iconSize, iconSize)
            ctx.globalCompositeOperation = 'source-atop'
            ctx.fillStyle = powerUpStyle.iconColor ?? powerUpStyle.shadowColor
            ctx.fillRect(x, y, iconSize, iconSize)
            ctx.globalCompositeOperation = 'source-over'
          } else if (!isGolden && !isTrail) {
            const highlightOffset = r * ORB_HIGHLIGHT_OFFSET
            const cx = drawX - highlightOffset
            const cy = drawY - highlightOffset
            if (powerUpStyle) {
              ctx.shadowColor = powerUpStyle.shadowColor
              ctx.shadowBlur = (ORB_GLOW_BLUR * 1.3 * sizeMult) / scale
              const orbGrad = ctx.createRadialGradient(cx, cy, 0, drawX, drawY, r)
              for (const [pos, color] of powerUpStyle.gradient) {
                orbGrad.addColorStop(pos, color)
              }
              ctx.fillStyle = orbGrad
            } else {
              ctx.shadowColor = 'rgba(255, 100, 0, 0.9)'
              ctx.shadowBlur = (ORB_GLOW_BLUR * valueMult * sizeMult) / scale
              const orbGrad = ctx.createRadialGradient(cx, cy, 0, drawX, drawY, r)
              orbGrad.addColorStop(0, 'rgba(255, 255, 255, 1)')
              orbGrad.addColorStop(0.2, 'rgba(255, 255, 120, 1)')
              orbGrad.addColorStop(0.45, 'rgba(255, 220, 60, 1)')
              orbGrad.addColorStop(0.7, 'rgba(255, 160, 40, 1)')
              orbGrad.addColorStop(0.9, 'rgba(255, 100, 80, 1)')
              orbGrad.addColorStop(1, 'rgba(220, 60, 100, 1)')
              ctx.fillStyle = orbGrad
            }
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.fill()
            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)'
            ctx.beginPath()
            ctx.arc(cx - r * ORB_SPECULAR_OFFSET, cy - r * ORB_SPECULAR_OFFSET, r * ORB_SPECULAR_RADIUS, 0, Math.PI * 2)
            ctx.fill()
            ctx.strokeStyle = isPowerUp ? 'rgba(255, 255, 255, 0.7)' : 'rgba(255, 200, 120, 0.6)'
            ctx.lineWidth = 1 / scale
            ctx.beginPath()
            ctx.arc(drawX, drawY, r, 0, Math.PI * 2)
            ctx.stroke()
          }
          ctx.restore()
        }
      }
      ctx.shadowColor = 'transparent'
      ctx.shadowBlur = 0

      const gameTime = state.gameTime ?? 0
      for (const snake of snakes) {
        const segs = snake.segments
        if (segs.length < 2) continue
        const ghostUntil = snake.ghostUntil ?? 0
        const isGhost = ghostUntil > gameTime
        const snakeColor = isGhost ? blendWithGhost(snake.color) : snake.color
        if (isGhost) ctx.save()
        if (isGhost) ctx.globalAlpha = 0.62
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        const bodyWidth = (BODY_RADIUS * 2) / scale
        const head = segs[0]
        ctx.save()
        ctx.shadowColor = snakeColor
        ctx.shadowBlur = (snake.isPlayer ? 14 : 10) / scale
        ctx.strokeStyle = snakeColor
        ctx.globalAlpha = snake.isPlayer ? 0.4 : 0.35
        ctx.lineWidth = bodyWidth + 4 / scale
        ctx.beginPath()
        strokeToroidalPath(ctx, segs, bounds)
        ctx.stroke()
        ctx.restore()
        ctx.beginPath()
        strokeToroidalPath(ctx, segs, bounds)
        const isPlayerBoost = snake.isPlayer && speedBoostActive && speedBoostProgress != null
        if (isPlayerBoost) {
          const pulse = 0.5 + 0.5 * Math.sin(speedBoostProgress * Math.PI * 6)
          ctx.save()
          ctx.shadowColor = snakeColor
          ctx.shadowBlur = (12 + pulse * 8) / scale
          ctx.strokeStyle = snakeColor
          ctx.lineWidth = bodyWidth + 10 / scale
          ctx.globalAlpha = 0.25 + pulse * 0.15
          ctx.stroke()
          ctx.restore()
          ctx.beginPath()
          strokeToroidalPath(ctx, segs, bounds)
        }
        ctx.strokeStyle = 'rgba(0,0,0,0.2)'
        ctx.lineWidth = bodyWidth + 6 / scale
        ctx.stroke()
        strokeToroidalPathWithGradient(
          ctx,
          segs,
          bounds,
          brightenColor(snakeColor, 0.35),
          snakeColor,
          bodyWidth,
        )
        ctx.fillStyle = snakeColor
        const headPositions =
          offset.x === 0 && offset.y === 0
            ? toroidalDrawPositions(head.x, head.y, bounds)
            : [[head.x, head.y]]
        for (const [hx, hy] of headPositions) {
          if (isPlayerBoost) {
            const bp = 0.5 + 0.5 * Math.sin(speedBoostProgress * Math.PI * 8)
            ctx.save()
            ctx.shadowColor = 'rgba(255, 230, 120, 0.9)'
            ctx.shadowBlur = (18 + bp * 14) / scale
            ctx.beginPath()
            ctx.arc(hx, hy, HEAD_RADIUS + 3 / scale, 0, Math.PI * 2)
            ctx.strokeStyle = `rgba(255, 220, 100, ${0.35 + bp * 0.35})`
            ctx.lineWidth = 3 / scale
            ctx.stroke()
            ctx.shadowBlur = 0
            ctx.restore()
          }
          ctx.beginPath()
          ctx.arc(hx, hy, HEAD_RADIUS, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = 'rgba(255,255,255,0.45)'
          ctx.lineWidth = 1.2 / scale
          ctx.stroke()
          ctx.fillStyle = 'rgba(255,255,255,0.35)'
          ctx.beginPath()
          ctx.arc(hx - HEAD_EYE_OFFSET, hy - HEAD_EYE_OFFSET, HEAD_RADIUS * 0.35, 0, Math.PI * 2)
          ctx.fill()
          ctx.fillStyle = 'rgba(255,255,255,0.25)'
          ctx.beginPath()
          ctx.arc(hx + HEAD_SPECULAR_OFFSET, hy + HEAD_SPECULAR_OFFSET, HEAD_SPECULAR_RADIUS, 0, Math.PI * 2)
          ctx.fill()
          ctx.fillStyle = snakeColor
        }
        if (isGhost) ctx.restore()
      }

      if (playerDeadSnake && deathAnimationProgress != null && deathAnimationProgress < 1) {
        drawDeathAnimation(ctx, playerDeadSnake, deathAnimationProgress, bounds, scale, offset)
      }
      const now = performance.now()
      for (const { snake, startTime } of botDeadSnakes) {
        const progress = Math.min(1, (now - startTime) / deathAnimMs)
        if (progress < 1) {
          drawDeathAnimation(ctx, snake, progress, bounds, scale, offset)
        }
      }

      ctx.restore()
    }

    ctx.restore()

    const minimapCanvas = minimapRef.current
    if (minimapCanvas && bounds) {
      const mW = MINIMAP_SIZE
      const mH = MINIMAP_SIZE
      if (minimapCanvas.width !== mW || minimapCanvas.height !== mH) {
        minimapCanvas.width = mW
        minimapCanvas.height = mH
      }
      const mCtx = minimapCanvas.getContext('2d')
      if (!mCtx) return
      const drawW = mW - MINIMAP_MARGIN * 2
      const drawH = mH - MINIMAP_MARGIN * 2
      const mScale = Math.min(drawW / bounds.width, drawH / bounds.height)
      const mOx = MINIMAP_MARGIN
      const mOy = MINIMAP_MARGIN
      mCtx.fillStyle = 'rgba(0,0,0,0.75)'
      mCtx.fillRect(0, 0, mW, mH)
      mCtx.strokeStyle = 'rgba(255,255,255,0.12)'
      mCtx.lineWidth = 1
      mCtx.strokeRect(mOx, mOy, bounds.width * mScale, bounds.height * mScale)
      mCtx.lineWidth = 10
      for (const p of pellets) {
        const pt = p.type ?? 'normal'
        if (pt === 'golden') {
          mCtx.fillStyle = 'rgba(255, 220, 80, 1)'
          mCtx.beginPath()
          mCtx.arc(mOx + p.x * mScale, mOy + p.y * mScale, 2.2, 0, Math.PI * 2)
          mCtx.fill()
        } else if (pt === 'trail') {
          mCtx.fillStyle = 'rgba(120, 200, 255, 0.75)'
          mCtx.beginPath()
          mCtx.arc(mOx + p.x * mScale, mOy + p.y * mScale, 0.7, 0, Math.PI * 2)
          mCtx.fill()
        } else {
          mCtx.fillStyle = 'rgba(255, 200, 60, 0.9)'
          mCtx.beginPath()
          mCtx.arc(mOx + p.x * mScale, mOy + p.y * mScale, 1, 0, Math.PI * 2)
          mCtx.fill()
        }
      }
      for (const snake of snakes) {
        const head = snake.segments[0]
        const mx = mOx + head.x * mScale
        const my = mOy + head.y * mScale
        const isPlayer = snake.isPlayer
        mCtx.fillStyle = snake.color
        mCtx.beginPath()
        mCtx.arc(mx, my, isPlayer ? 3 : 2, 0, Math.PI * 2)
        mCtx.fill()
        if (isPlayer) {
          mCtx.strokeStyle = 'rgba(255,255,255,0.9)'
          mCtx.lineWidth = 1
          mCtx.beginPath()
          mCtx.arc(mx, my, 4, 0, Math.PI * 2)
          mCtx.stroke()
        }
      }
      const { scale, camX, camY } = cameraRef.current
      const vw = canvas.width / scale
      const vh = canvas.height / scale
      const vx = mOx + (-camX / scale) * mScale
      const vy = mOy + (-camY / scale) * mScale
      mCtx.strokeStyle = 'rgba(255,255,255,0.35)'
      mCtx.lineWidth = 1
      mCtx.strokeRect(vx, vy, vw * mScale, vh * mScale)
    }
  }, [
    state,
    state?.bounds,
    resizeTick,
    playerDeadSnake,
    deathAnimationProgress,
    botDeadSnakes,
    deathAnimMs,
    speedBoostActive,
    speedBoostProgress,
    powerupImagesReady,
    cameraShakeX,
    cameraShakeY,
  ])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !onCanvasBoost) return
    const onDown = (e) => {
      if (e.button === 0) onCanvasBoost()
    }
    canvas.addEventListener('mousedown', onDown)
    canvas.addEventListener('touchstart', onDown, { passive: true })
    return () => {
      canvas.removeEventListener('mousedown', onDown)
      canvas.removeEventListener('touchstart', onDown)
    }
  }, [onCanvasBoost])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !onMouseMove) return
    const handleMove = (e) => {
      const { scale, camX, camY } = cameraRef.current
      if (scale <= 0 || canvas.width <= 0 || canvas.height <= 0) return
      const rect = canvas.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return
      const canvasX = ((e.clientX - rect.left) / rect.width) * canvas.width
      const canvasY = ((e.clientY - rect.top) / rect.height) * canvas.height
      const worldX = (canvasX - camX) / scale
      const worldY = (canvasY - camY) / scale
      onMouseMove(worldX, worldY)
    }
    canvas.addEventListener('mousemove', handleMove)
    return () => canvas.removeEventListener('mousemove', handleMove)
  }, [onMouseMove])

  return (
    <div ref={wrapRef} className="slither-canvas-wrap">
      <canvas
        ref={canvasRef}
        className="slither-canvas"
        aria-label="Slither game view. Click to speed boost."
      />
      <div className="slither-minimap-wrap">
        <canvas
          ref={minimapRef}
          className="slither-minimap"
          aria-label="Minimap of game area"
        />
      </div>
    </div>
  )
}
