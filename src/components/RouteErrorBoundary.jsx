import { Component } from 'react';

export default class RouteErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, routeKey: props.routeKey };
  }

  static getDerivedStateFromProps(props, state) {
    if (props.routeKey !== state.routeKey) return { error: null, routeKey: props.routeKey };
    return null;
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, details) {
    console.error('Polymath section render failed:', error, details);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <section className="route-error-state" role="alert">
        <span aria-hidden="true">↻</span>
        <h1>This section did not open correctly.</h1>
        <p>Your account and work are safe. Reload this section or return to the piano.</p>
        <div>
          <button className="primary" type="button" onClick={() => window.location.reload()}>Reload section</button>
          <button className="ghost" type="button" onClick={() => this.props.onNavigate?.('studio')}>Back to piano</button>
        </div>
      </section>
    );
  }
}
