/**
 * Slither-style game logic: continuous movement, collision, pellets, death.
 * Multiplayer-ready: state shape and tick() can be driven by server later.
 */

export const SEGMENT_SPACING = 8
export const HEAD_RADIUS = 15
export const BODY_RADIUS = 15
/** Extra distance for death (collision with other snakes) so death feels readable. */
export const DEATH_RADIUS_EXTRA = 12
export const PELLET_RADIUS = 30
/** Extra radius for pellet collection (magnet effect). */
export const MAGNET_RADIUS = 155
export const DEFAULT_SPEED = 1000
/** Speed multiplier per segment over initial length (e.g. 0.006 = 0.6% per pellet). */
export const SPEED_GROWTH_PER_SEGMENT = 0.006
/** Max speed multiplier from growth (cap so long snakes don't go crazy). */
export const SPEED_GROWTH_CAP = 1.6
export const TURN_SPEED = 6
export const PELLET_VALUE = 2
export const DEATH_PELLET_FRACTION = 0.4
export const INITIAL_LENGTH = 80
export const PELLET_SPAWN_INTERVAL = 2
export const ARENA_PADDING = 40

/** Power-up pellet types. */
export const POWERUP_TYPES = /** @type {const} */ (['shield', 'ghost', 'magnet'])
/** Probability per spawn that a new pellet is a power-up (0–1). */
export const POWERUP_SPAWN_CHANCE = 0.16
/** Duration in seconds for ghost and magnet effects. */
export const GHOST_DURATION = 4
export const MAGNET_DURATION = 5
/** Magnet radius multiplier when super-magnet power-up is active. */
export const SUPER_MAGNET_MULTIPLIER = 2
/** Segments shed per second while boosting (Slither.io-style mass trail). */
export const BOOST_SHED_SEGMENTS_PER_SEC = 7
/** Do not shrink player below this many segments while boosting. */
export const MIN_PLAYER_SEGMENTS_FOR_BOOST_DRAIN = 42
/** Rare high-value pellet (like Slither “special” food). */
export const GOLDEN_PELLET_VALUE = 12
/** Chance a spawned normal-cycle pellet is golden instead of regular. */
export const GOLDEN_SPAWN_CHANCE = 0.045

/**
 * @typedef {{ x: number, y: number }} Point
 * @typedef {'normal' | 'shield' | 'ghost' | 'magnet' | 'trail' | 'golden'} PelletType
 * @typedef {{ id: string, segments: Point[], angle: number, speed: number, turnSpeed: number, color: string, isPlayer?: boolean, shield?: boolean, ghostUntil?: number, magnetUntil?: number }} Snake
 * @typedef {{ x: number, y: number, value: number, type?: PelletType }} Pellet
 * @typedef {{ width: number, height: number }} Bounds
 * @typedef {{ snakes: Snake[], pellets: Pellet[], bounds: Bounds, nextSnakeId: number, nextPelletSpawn: number, gameTime?: number, playerBoostShedAcc?: number }} GameState
 */

/**
 * Normalize angle to [-PI, PI].
 * @param {number} a
 * @returns {number}
 */
function normalizeAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI
  while (a < -Math.PI) a += 2 * Math.PI
  return a
}

/**
 * Create initial game state.
 * @param {{ bounds?: Bounds, numBots?: number, numPellets?: number }} options
 * @returns {GameState}
 */
export function createInitialState(options = {}) {
  const bounds = options.bounds ?? { width: 28000, height: 28000 }
  const numBots = options.numBots ?? 8
  const numPellets = options.numPellets ?? 80
  const padding = ARENA_PADDING
  const minX = padding
  const maxX = bounds.width - padding
  const minY = padding
  const maxY = bounds.height - padding

  const colors = [
    '#33cc33',
    '#e74c3c',
    '#3498db',
    '#f1c40f',
    '#9b59b6',
    '#1abc9c',
    '#e67e22',
    '#2ecc71',
  ]

  const snakes = []
  const usedPositions = new Set()
  const playerColor = '#00ff88'

  for (let i = 0; i < numBots; i++) {
    let x, y, key
    do {
      x = minX + Math.random() * (maxX - minX)
      y = minY + Math.random() * (maxY - minY)
      key = `${Math.floor(x / 50)},${Math.floor(y / 50)}`
    } while (usedPositions.has(key))
    usedPositions.add(key)

    const angle = Math.random() * 2 * Math.PI - Math.PI
    const segments = []
    for (let s = 0; s < INITIAL_LENGTH; s++) {
      segments.push({
        x: x - Math.cos(angle) * s * SEGMENT_SPACING,
        y: y - Math.sin(angle) * s * SEGMENT_SPACING,
      })
    }

    const isPlayer = i === 0
    snakes.push({
      id: isPlayer ? 'player' : `snake-${i}`,
      segments,
      angle,
      speed: DEFAULT_SPEED,
      turnSpeed: TURN_SPEED,
      color: isPlayer ? playerColor : colors[i % colors.length],
      isPlayer,
      shield: false,
      ghostUntil: 0,
      magnetUntil: 0,
    })
  }

  const pellets = []
  for (let p = 0; p < numPellets; p++) {
    pellets.push({
      x: minX + Math.random() * (maxX - minX),
      y: minY + Math.random() * (maxY - minY),
      value: PELLET_VALUE,
      type: 'normal',
    })
  }

  return {
    snakes,
    pellets,
    bounds,
    nextSnakeId: numBots,
    nextPelletSpawn: PELLET_SPAWN_INTERVAL,
    gameTime: 0,
    playerBoostShedAcc: 0,
  }
}

/**
 * Wrap a point into [0, bounds.width) x [0, bounds.height).
 * @param {Point} p
 * @param {Bounds} bounds
 * @returns {Point}
 */
export function wrapPoint(p, bounds) {
  const w = bounds.width
  const h = bounds.height
  return {
    x: ((p.x % w) + w) % w,
    y: ((p.y % h) + h) % h,
  }
}

/**
 * Shortest distance squared between two points on a toroidal arena.
 * @param {Point} a
 * @param {Point} b
 * @param {Bounds} bounds
 * @returns {number}
 */
export function toroidalDistSq(a, b, bounds) {
  let dx = b.x - a.x
  let dy = b.y - a.y
  dx = dx - bounds.width * Math.round(dx / bounds.width)
  dy = dy - bounds.height * Math.round(dy / bounds.height)
  return dx * dx + dy * dy
}

/**
 * Move snake head and update body segments (follow-the-leader).
 * Positions are wrapped into bounds for toroidal arena.
 * @param {Snake} snake
 * @param {number} dt in seconds
 * @param {number} targetAngle desired angle (interpolated by turnSpeed)
 * @param {Bounds} bounds
 * @returns {Snake}
 */
function moveSnake(snake, dt, targetAngle, bounds) {
  const head = snake.segments[0]
  let angle = snake.angle
  const diff = normalizeAngle(targetAngle - angle)
  angle += diff * Math.min(1, snake.turnSpeed * dt)
  angle = normalizeAngle(angle)

  const extraSegments = Math.max(0, snake.segments.length - INITIAL_LENGTH)
  const growthMul = Math.min(SPEED_GROWTH_CAP, 1 + extraSegments * SPEED_GROWTH_PER_SEGMENT)
  const effectiveSpeed = snake.speed * growthMul
  const dx = Math.cos(angle) * effectiveSpeed * dt
  const dy = Math.sin(angle) * effectiveSpeed * dt
  let newHead = wrapPoint({ x: head.x + dx, y: head.y + dy }, bounds)

  const w = bounds.width
  const h = bounds.height
  const segments = [newHead]
  let prev = newHead
  for (let i = 1; i < snake.segments.length; i++) {
    const curr = snake.segments[i]
    // Use toroidal (shortest) direction so segments follow the wrap and don't jump to the head's side.
    let tdx = curr.x - prev.x
    let tdy = curr.y - prev.y
    tdx = tdx - w * Math.round(tdx / w)
    tdy = tdy - h * Math.round(tdy / h)
    const d = Math.hypot(tdx, tdy)
    let next
    if (d <= 1e-6) {
      next = {
        x: prev.x - Math.cos(angle) * SEGMENT_SPACING,
        y: prev.y - Math.sin(angle) * SEGMENT_SPACING,
      }
    } else {
      next = {
        x: prev.x + (tdx / d) * SEGMENT_SPACING,
        y: prev.y + (tdy / d) * SEGMENT_SPACING,
      }
    }
    const wrappedNext = wrapPoint(next, bounds)
    segments.push(wrappedNext)
    prev = segments[segments.length - 1]
  }

  return {
    ...snake,
    segments,
    angle,
  }
}

/**
 * Check if head hits another snake's body (not own body). Uses toroidal distance.
 * Skips collision if victim or body-owner has ghost active at gameTime.
 * @param {string} snakeId
 * @param {Point} head
 * @param {Snake[]} allSnakes
 * @param {Bounds} bounds
 * @param {number} gameTime
 * @returns {boolean}
 */
function headHitsBody(snakeId, head, allSnakes, bounds, gameTime) {
  const contactDist = HEAD_RADIUS + BODY_RADIUS + DEATH_RADIUS_EXTRA
  const contactDistSq = contactDist * contactDist
  const victim = allSnakes.find((s) => s.id === snakeId)
  const victimGhost = victim && (victim.ghostUntil ?? 0) > gameTime
  for (const snake of allSnakes) {
    if (snake.id === snakeId) continue
    const ownerGhost = (snake.ghostUntil ?? 0) > gameTime
    if (victimGhost || ownerGhost) continue
    for (let i = 0; i < snake.segments.length; i++) {
      if (toroidalDistSq(head, snake.segments[i], bounds) <= contactDistSq) return true
    }
  }
  return false
}

/**
 * While boosting, drop trail pellets from the tail and shrink the player (Slither.io-style).
 * @param {Snake[]} snakes
 * @param {Pellet[]} pellets
 * @param {number} dt
 * @param {number} playerSpeedMultiplier
 * @param {number} boostShedAcc
 * @returns {{ snakes: Snake[], pellets: Pellet[], boostShedAcc: number }}
 */
function applyPlayerBoostDrain(snakes, pellets, dt, playerSpeedMultiplier, boostShedAcc) {
  if (playerSpeedMultiplier <= 1) {
    return { snakes, pellets, boostShedAcc }
  }
  let acc = (boostShedAcc ?? 0) + dt * BOOST_SHED_SEGMENTS_PER_SEC
  const playerIdx = snakes.findIndex((s) => s.isPlayer)
  if (playerIdx < 0) {
    return { snakes, pellets, boostShedAcc: acc }
  }
  let player = snakes[playerIdx]
  const nextPellets = [...pellets]
  const nextSnakes = [...snakes]
  while (acc >= 1 && player.segments.length > MIN_PLAYER_SEGMENTS_FOR_BOOST_DRAIN) {
    acc -= 1
    const tail = player.segments[player.segments.length - 1]
    nextPellets.push({
      x: tail.x,
      y: tail.y,
      value: 1,
      type: 'trail',
    })
    player = { ...player, segments: player.segments.slice(0, -1) }
  }
  nextSnakes[playerIdx] = player
  return { snakes: nextSnakes, pellets: nextPellets, boostShedAcc: acc }
}

/**
 * Apply target angles from bot AI and move all snakes.
 * @param {GameState} state
 * @param {number} dt
 * @param {Record<string, number>} targetAngles snakeId -> target angle in radians
 * @param {{ playerSpeedMultiplier?: number }} [options]
 * @returns {{ state: GameState, deadIds: string[] }}
 */
export function tick(state, dt, targetAngles = {}, options = {}) {
  const { bounds } = state
  let { snakes, pellets } = state
  const gameTime = (state.gameTime ?? 0) + dt
  const padding = ARENA_PADDING
  const minX = padding
  const maxX = bounds.width - padding
  const minY = padding
  const maxY = bounds.height - padding
  const playerSpeedMultiplier = options.playerSpeedMultiplier ?? 1

  snakes = snakes.map((s) => {
    const target = targetAngles[s.id] ?? s.angle
    const speedMul = s.isPlayer ? playerSpeedMultiplier : 1
    const snakeToMove = speedMul !== 1 ? { ...s, speed: s.speed * speedMul } : s
    const result = moveSnake(snakeToMove, dt, target, bounds)
    if (speedMul !== 1) return { ...result, speed: s.speed }
    return result
  })

  const drained = applyPlayerBoostDrain(
    snakes,
    pellets,
    dt,
    playerSpeedMultiplier,
    state.playerBoostShedAcc ?? 0,
  )
  snakes = drained.snakes
  pellets = drained.pellets
  const playerBoostShedAcc = drained.boostShedAcc

  const baseCollectRadius = HEAD_RADIUS + PELLET_RADIUS + MAGNET_RADIUS
  const collectedIndices = new Set()
  snakes = snakes.map((snake) => {
    const head = snake.segments[0]
    const hasMagnet = (snake.magnetUntil ?? 0) > gameTime
    const collectRadius = hasMagnet ? baseCollectRadius * SUPER_MAGNET_MULTIPLIER : baseCollectRadius
    const pelletCollectRadiusSq = collectRadius * collectRadius
    let lengthGain = 0
    let shield = snake.shield ?? false
    let ghostUntil = snake.ghostUntil ?? 0
    let magnetUntil = snake.magnetUntil ?? 0
    for (let i = 0; i < pellets.length; i++) {
      if (collectedIndices.has(i)) continue
      if (toroidalDistSq(head, pellets[i], bounds) >= pelletCollectRadiusSq) continue
      collectedIndices.add(i)
      const pellet = pellets[i]
      const ptype = pellet.type ?? 'normal'
      if (ptype === 'shield') shield = true
      else if (ptype === 'ghost') ghostUntil = gameTime + GHOST_DURATION
      else if (ptype === 'magnet') magnetUntil = gameTime + MAGNET_DURATION
      else if (ptype === 'normal' || ptype === 'trail' || ptype === 'golden') {
        lengthGain += pellet.value
      }
    }
    const needsUpdate =
      lengthGain > 0 ||
      shield !== (snake.shield ?? false) ||
      ghostUntil !== (snake.ghostUntil ?? 0) ||
      magnetUntil !== (snake.magnetUntil ?? 0)
    if (!needsUpdate) return snake
    let nextSnake = { ...snake, shield, ghostUntil, magnetUntil }
    if (lengthGain > 0) {
      const tail = nextSnake.segments[nextSnake.segments.length - 1]
      const newSegments = [...nextSnake.segments]
      for (let g = 0; g < lengthGain; g++) {
        newSegments.push({ ...tail })
      }
      nextSnake = { ...nextSnake, segments: newSegments }
    }
    return nextSnake
  })
  pellets = pellets.filter((_, i) => !collectedIndices.has(i))

  const deadIds = new Set()
  for (const snake of snakes) {
    const head = snake.segments[0]
    if (headHitsBody(snake.id, head, snakes, bounds, gameTime)) {
      deadIds.add(snake.id)
    }
  }

  snakes = snakes.map((s) => {
    if (deadIds.has(s.id) && (s.shield === true)) {
      deadIds.delete(s.id)
      return { ...s, shield: false }
    }
    return s
  })

  snakes = snakes.filter((s) => !deadIds.has(s.id))

  for (const id of deadIds) {
    const dead = state.snakes.find((s) => s.id === id)
    if (!dead) continue
    const count = Math.max(1, Math.floor(dead.segments.length * DEATH_PELLET_FRACTION))
    const step = Math.max(1, Math.floor(dead.segments.length / count))
    for (let i = 0; i < count; i++) {
      const idx = Math.min(i * step, dead.segments.length - 1)
      const seg = dead.segments[idx]
      pellets.push({
        x: seg.x,
        y: seg.y,
        value: PELLET_VALUE,
        type: 'normal',
      })
    }
  }

  let { nextPelletSpawn } = state
  nextPelletSpawn -= dt
  if (nextPelletSpawn <= 0) {
    nextPelletSpawn = PELLET_SPAWN_INTERVAL
    const roll = Math.random()
    let type
    let value
    if (roll < POWERUP_SPAWN_CHANCE) {
      type = POWERUP_TYPES[Math.floor(Math.random() * POWERUP_TYPES.length)]
      value = 1
    } else if (roll < POWERUP_SPAWN_CHANCE + GOLDEN_SPAWN_CHANCE) {
      type = 'golden'
      value = GOLDEN_PELLET_VALUE
    } else {
      type = 'normal'
      value = PELLET_VALUE
    }
    pellets.push({
      x: minX + Math.random() * (maxX - minX),
      y: minY + Math.random() * (maxY - minY),
      value,
      type,
    })
  }

  return {
    state: {
      ...state,
      snakes,
      pellets,
      nextPelletSpawn,
      gameTime,
      playerBoostShedAcc,
    },
    deadIds: [...deadIds],
  }
}

/**
 * Get snake length (segment count) for display.
 * @param {Snake} snake
 * @returns {number}
 */
export function getSnakeLength(snake) {
  return snake.segments.length
}
