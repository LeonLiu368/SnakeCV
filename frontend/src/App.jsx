import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import './App.css'

const GRID_SIZE = 18
const START_SPEED = 160
const DIRECTIONS = {
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  w: { x: 0, y: -1 },
  s: { x: 0, y: 1 },
  a: { x: -1, y: 0 },
  d: { x: 1, y: 0 },
}
const HEAD_DIRECTIONS = {
  UP: { x: 0, y: -1 },
  DOWN: { x: 0, y: 1 },
  LEFT: { x: -1, y: 0 },
  RIGHT: { x: 1, y: 0 },
}
const NOSE_THRESHOLD = 0.05
const NOSE_INDEX = 1

const randomFood = (snake) => {
  const occupied = new Set(snake.map((seg) => `${seg.x},${seg.y}`))
  let spot = null
  while (!spot || occupied.has(`${spot.x},${spot.y}`)) {
    spot = {
      x: Math.floor(Math.random() * GRID_SIZE),
      y: Math.floor(Math.random() * GRID_SIZE),
    }
  }
  return spot
}

function App() {
  const [snake, setSnake] = useState([
    { x: 6, y: 9 },
    { x: 5, y: 9 },
    { x: 4, y: 9 },
  ])
  const [food, setFood] = useState(() => randomFood([{ x: 6, y: 9 }]))
  const [direction, setDirection] = useState({ x: 1, y: 0 })
  const [running, setRunning] = useState(false)
  const [score, setScore] = useState(0)
  const [best, setBest] = useState(0)
  const [status, setStatus] = useState('Press Start')
  const [cameraStatus, setCameraStatus] = useState('Initializing camera…')
  const [headDirection, setHeadDirection] = useState(null)
  const queuedDirection = useRef(direction)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const boardCells = useMemo(
    () => Array.from({ length: GRID_SIZE * GRID_SIZE }),
    []
  )

  const reset = useCallback(() => {
    const freshSnake = [
      { x: 6, y: 9 },
      { x: 5, y: 9 },
      { x: 4, y: 9 },
    ]
    setSnake(freshSnake)
    setFood(randomFood(freshSnake))
    setDirection({ x: 1, y: 0 })
    queuedDirection.current = { x: 1, y: 0 }
    setScore(0)
    setStatus('Ready')
  }, [])

  useEffect(() => {
    const handleKey = (event) => {
      const next = DIRECTIONS[event.key]
      if (!next) return
      if (direction.x + next.x === 0 && direction.y + next.y === 0) return
      queuedDirection.current = next
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [direction])

  useEffect(() => {
    let active = true
    let landmarker = null
    let animationId = null

    const setup = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        )
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
        })

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        })
        if (!videoRef.current) return
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        videoRef.current.style.transform = 'scaleX(-1)'
        setCameraStatus('Head tracking active')

        const loop = () => {
          if (!active || !videoRef.current) return
          if (videoRef.current.readyState >= 2 && landmarker) {
            const canvas = canvasRef.current
            if (canvas && videoRef.current.videoWidth && videoRef.current.videoHeight) {
              canvas.width = videoRef.current.videoWidth
              canvas.height = videoRef.current.videoHeight
            }
            const result = landmarker.detectForVideo(
              videoRef.current,
              performance.now()
            )
            if (result.faceLandmarks && result.faceLandmarks.length) {
              const nose = result.faceLandmarks[0][NOSE_INDEX]
              const direction = noseDirection(nose)
              const mirrored =
                direction === 'LEFT' ? 'RIGHT' : direction === 'RIGHT' ? 'LEFT' : direction
              setHeadDirection(mirrored)
              drawOverlay(nose, mirrored, true)
              setCameraStatus('Face detected')
              if (direction) {
                const next = HEAD_DIRECTIONS[mirrored]
                const current = queuedDirection.current
                if (!(current.x + next.x === 0 && current.y + next.y === 0)) {
                  queuedDirection.current = next
                }
              }
            } else {
              setHeadDirection(null)
              clearOverlay()
              setCameraStatus('No face detected')
            }
          }
          animationId = requestAnimationFrame(loop)
        }

        animationId = requestAnimationFrame(loop)
      } catch (error) {
        console.error(error)
        setCameraStatus('Camera access failed')
      }
    }

    const noseDirection = (nose) => {
      const dx = nose.x - 0.5
      const dy = nose.y - 0.5
      if (Math.abs(dx) < NOSE_THRESHOLD && Math.abs(dy) < NOSE_THRESHOLD) {
        return null
      }
      if (Math.abs(dx) > Math.abs(dy)) {
        return dx > 0 ? 'RIGHT' : 'LEFT'
      }
      return dy > 0 ? 'DOWN' : 'UP'
    }

    const clearOverlay = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }

    const drawOverlay = (nose, direction, mirror) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const cx = (mirror ? 1 - nose.x : nose.x) * canvas.width
      const cy = nose.y * canvas.height
      ctx.fillStyle = 'rgba(126, 240, 193, 0.9)'
      ctx.beginPath()
      ctx.arc(cx, cy, 6, 0, Math.PI * 2)
      ctx.fill()
      if (!direction) return
      const arrowLen = 120
      let ex = cx
      let ey = cy
      if (direction === 'RIGHT') ex += arrowLen
      if (direction === 'LEFT') ex -= arrowLen
      if (direction === 'UP') ey -= arrowLen
      if (direction === 'DOWN') ey += arrowLen
      ctx.strokeStyle = 'rgba(255, 211, 106, 0.95)'
      ctx.lineWidth = 4
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(ex, ey)
      ctx.stroke()
      ctx.fillStyle = 'rgba(255, 211, 106, 0.95)'
      ctx.font = '18px Manrope, sans-serif'
      ctx.fillText(direction, cx - 24, cy + 28)
    }

    setup()

    return () => {
      active = false
      if (animationId) {
        cancelAnimationFrame(animationId)
      }
      setHeadDirection(null)
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  useEffect(() => {
    if (!running) return undefined
    const interval = setInterval(() => {
      setSnake((prev) => {
        const nextDirection = queuedDirection.current
        setDirection(nextDirection)
        const head = prev[0]
        const nextHead = {
          x: head.x + nextDirection.x,
          y: head.y + nextDirection.y,
        }
        const hitWall =
          nextHead.x < 0 ||
          nextHead.y < 0 ||
          nextHead.x >= GRID_SIZE ||
          nextHead.y >= GRID_SIZE
        const hitSelf = prev.some(
          (seg) => seg.x === nextHead.x && seg.y === nextHead.y
        )
        if (hitWall || hitSelf) {
          setRunning(false)
          setStatus('Game Over')
          setBest((current) => Math.max(current, score))
          return prev
        }
        const nextSnake = [nextHead, ...prev]
        if (nextHead.x === food.x && nextHead.y === food.y) {
          const nextScore = score + 10
          setScore(nextScore)
          setFood(randomFood(nextSnake))
          return nextSnake
        }
        nextSnake.pop()
        return nextSnake
      })
    }, START_SPEED)
    return () => clearInterval(interval)
  }, [food, running, score])

  const handleStart = () => {
    if (!running) {
      if (status === 'Game Over') {
        reset()
      }
      setRunning(true)
      setStatus('Running')
    }
  }

  const handlePause = () => {
    setRunning(false)
    setStatus('Paused')
  }

  return (
    <div className="app">
      <header className="hud">
        <div className="title">
          <p className="eyebrow">Neon Snake</p>
          <h1>{status}</h1>
        </div>
        <div className="stats">
          <div>
            <p className="label">Score</p>
            <p className="value">{score}</p>
          </div>
          <div>
            <p className="label">Best</p>
            <p className="value">{best}</p>
          </div>
        </div>
        <div className="actions">
          <button className="primary" onClick={handleStart}>
            Start
          </button>
          <button className="ghost" onClick={handlePause}>
            Pause
          </button>
        </div>
      </header>

      <main className="arena">
        <div className="board playable">
          <div className="grid" />
          <div className="board-cells">
            {boardCells.map((_, idx) => {
              const x = idx % GRID_SIZE
              const y = Math.floor(idx / GRID_SIZE)
              const isSnake = snake.some((seg) => seg.x === x && seg.y === y)
              const isHead = snake[0].x === x && snake[0].y === y
              const isFood = food.x === x && food.y === y
              const className = [
                'cell',
                isSnake ? 'snake' : '',
                isHead ? 'head' : '',
                isFood ? 'food' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return <span key={idx} className={className} />
            })}
          </div>
        </div>

        <div className="camera-panel">
          <div className="camera-frame">
            <video ref={videoRef} muted playsInline />
            <canvas ref={canvasRef} />
            <p className="camera-status">{cameraStatus}</p>
            {headDirection ? (
              <p className="camera-direction">{headDirection}</p>
            ) : null}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
