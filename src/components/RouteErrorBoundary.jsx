import { Component } from 'react';
import RouteLoadingProgress from './RouteLoadingProgress.jsx';
import { isRetryableRouteLoadError } from '../utils/routeLoading.js';

const ROUTE_RECOVERY_COOLDOWN_MS = 3 * 60 * 1000;
const ROUTE_RECOVERY_DELAY_MS = 6000;

function recoveryStorageKey(routeKey) {
  return `polymath-route-recovery:${String(routeKey || 'unknown')}`;
}

function canAttemptAutomaticRecovery(routeKey) {
  try {
    const previous = Number(window.sessionStorage.getItem(recoveryStorageKey(routeKey)) || 0);
    return !previous || Date.now() - previous > ROUTE_RECOVERY_COOLDOWN_MS;
  } catch {
    return false;
  }
}

function rememberAutomaticRecovery(routeKey) {
  try {
    window.sessionStorage.setItem(recoveryStorageKey(routeKey), String(Date.now()));
  } catch {
    // The caller already skips automatic refresh when storage is unavailable.
  }
}

export default class RouteErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, routeKey: props.routeKey, recovering: false };
    this.recoveryTimer = null;
  }

  static getDerivedStateFromProps(props, state) {
    if (props.routeKey !== state.routeKey) {
      return { error: null, routeKey: props.routeKey, recovering: false };
    }
    return null;
  }

  static getDerivedStateFromError(error) {
    return { error, recovering: isRetryableRouteLoadError(error) };
  }

  componentDidCatch(error, details) {
    console.error('Polymath section render failed:', error, details);
    if (!isRetryableRouteLoadError(error)) return;
    if (!canAttemptAutomaticRecovery(this.props.routeKey)) {
      this.setState({ recovering: false });
      return;
    }
    rememberAutomaticRecovery(this.props.routeKey);
    this.recoveryTimer = window.setTimeout(
      () => window.location.reload(),
      ROUTE_RECOVERY_DELAY_MS,
    );
  }

  componentDidUpdate(previousProps) {
    if (previousProps.routeKey === this.props.routeKey) return;
    if (this.recoveryTimer) window.clearTimeout(this.recoveryTimer);
    this.recoveryTimer = null;
  }

  componentWillUnmount() {
    if (this.recoveryTimer) window.clearTimeout(this.recoveryTimer);
  }

  render() {
    if (!this.state.error) return this.props.children;
    if (this.state.recovering) {
      return (
        <section className="route-loading route-recovery-loading" aria-live="polite">
          <RouteLoadingProgress
            label="Still opening this section..."
            initialElapsedMs={30000}
          />
        </section>
      );
    }
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
