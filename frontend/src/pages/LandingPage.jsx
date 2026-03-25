import { Link } from 'react-router-dom'

const GAMES = [
  {
    id: 'snake',
    title: 'SnakeCV',
    description: 'Classic Snake controlled by your face. Use head movement or keyboard to steer. Play with your camera on.',
    to: '/games/snake',
    cta: 'Play SnakeCV',
    thumbGradient: 'linear-gradient(135deg, #1a472a 0%, #0d2818 50%, #0d0d0d 100%)',
    thumbLabel: 'S',
  },
  {
    id: 'slither',
    title: 'Slither',
    description: 'Continuous snake: grow by eating pellets, avoid bodies and walls. Face or mouse control.',
    to: '/games/slither',
    cta: 'Play Slither',
    thumbGradient: 'linear-gradient(135deg, #2a1a4a 0%, #1a0d2e 50%, #0d0d0d 100%)',
    thumbLabel: 'Sl',
  },
]

export function LandingPage() {
  return (
    <div className="landing">
      <div className="landing-atmosphere" aria-hidden="true">
        <span className="landing-bubble landing-bubble--1" />
        <span className="landing-bubble landing-bubble--2" />
        <span className="landing-bubble landing-bubble--3" />
        <span className="landing-bubble landing-bubble--4" />
      </div>
      <header className="landing-header">
        <Link to="/" className="landing-brand-link">
          <h1 className="landing-brand">CVified</h1>
        </Link>
        <nav className="landing-nav" aria-label="Main">
          <Link to="/" className="landing-nav-link">Home</Link>
        </nav>
      </header>
      <main className="landing-main">
        <p className="landing-tagline">Computer vision games in the browser.</p>
        <section className="landing-section" aria-labelledby="games-heading">
          <h2 id="games-heading" className="landing-section-title">Games</h2>
          <ul className="game-list game-grid" aria-label="Games">
            {GAMES.map((game) => (
              <li key={game.id}>
                <article className="game-card">
                  <div
                    className="game-card-thumb"
                    style={{ background: game.thumbGradient }}
                  >
                    <span className="game-card-thumb-label" aria-hidden>{game.thumbLabel}</span>
                  </div>
                  <h3 className="game-card-title">{game.title}</h3>
                  <p className="game-card-desc">{game.description}</p>
                  <Link to={game.to} className="game-card-cta primary">
                    {game.cta}
                  </Link>
                </article>
              </li>
            ))}
          </ul>
        </section>
      </main>
      <footer className="landing-footer">
        <span>CVified – computer vision games</span>
      </footer>
    </div>
  )
}
