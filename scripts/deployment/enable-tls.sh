#!/usr/bin/env bash
# =============================================================================
# Obtain a Let's Encrypt certificate and switch nginx over to HTTPS.
# =============================================================================
# Run once on the deployment host, from the repository root:
#
#     ./scripts/deployment/enable-tls.sh
#
# Requires DOMAIN and CERTBOT_EMAIL in .env, a public DNS A/AAAA record for
# DOMAIN pointing at this host, and inbound port 80 reachable from the internet
# (Let's Encrypt validates by fetching a file over plain HTTP).
#
# Ordering matters and is the reason this is a script rather than a compose
# flag: nginx will not start with an ssl_certificate that does not exist yet,
# and certbot cannot answer the challenge unless nginx is already serving :80.
# So the HTTP-only config runs first, the certificate is issued through it, and
# only then is the TLS config installed.
#
# Re-running is safe: certbot will not reissue a certificate that is still
# valid unless --force-renewal is passed.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

COMPOSE="docker compose -f docker-compose.prod.yml"
NGINX_DIR="deployment/nginx"
TEMPLATE="$NGINX_DIR/tls-available/ml-monitor-tls.conf.template"
LIVE_CONF="$NGINX_DIR/conf.d/ml-monitor.conf"
BACKUP_CONF="$NGINX_DIR/tls-available/ml-monitor-http.conf.bak"

log() { printf '\033[0;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Configuration ------------------------------------------------------
[ -f .env ] || die ".env not found. Copy .env.example and fill it in first."
# shellcheck disable=SC1091
set -a; . ./.env; set +a

: "${DOMAIN:?DOMAIN must be set in .env (e.g. DOMAIN=monitor.example.com)}"
: "${CERTBOT_EMAIL:?CERTBOT_EMAIL must be set in .env (used for expiry notices)}"

STAGING_FLAG=""
if [ "${CERTBOT_STAGING:-false}" = "true" ]; then
    # Let's Encrypt rate-limits failed issuance hard (5 failures per account,
    # per hostname, per hour). Rehearse against staging before going live.
    STAGING_FLAG="--staging"
    log "Using the Let's Encrypt STAGING environment; the certificate will NOT be trusted by browsers."
fi

log "Domain: $DOMAIN"
log "Contact: $CERTBOT_EMAIL"

# --- 2. Serve :80 so the challenge can be answered -------------------------
log "Starting nginx on port 80 (HTTP-only) so certbot can be validated..."
$COMPOSE up -d nginx
until curl -sf -o /dev/null "http://localhost/health"; do
    printf '.'
    sleep 2
done
printf '\n'
log "nginx is answering on :80."

# --- 3. Request the certificate --------------------------------------------
log "Requesting a certificate from Let's Encrypt..."
# shellcheck disable=SC2086
$COMPOSE run --rm certbot certonly \
    --webroot --webroot-path=/var/www/certbot \
    --email "$CERTBOT_EMAIL" \
    --agree-tos --no-eff-email \
    --non-interactive \
    $STAGING_FLAG \
    -d "$DOMAIN" \
    || die "Certificate issuance failed. Check that $DOMAIN resolves to this host and that port 80 is reachable from the internet."

CERT_PATH="$NGINX_DIR/letsencrypt/live/$DOMAIN/fullchain.pem"
[ -f "$CERT_PATH" ] || die "certbot reported success but $CERT_PATH is missing."
log "Certificate issued: $CERT_PATH"

# --- 4. Install the TLS config ---------------------------------------------
log "Installing the TLS site configuration..."
[ -f "$BACKUP_CONF" ] || cp "$LIVE_CONF" "$BACKUP_CONF"

# Only DOMAIN is substituted; every $host/$scheme nginx variable is preserved.
sed "s|\${DOMAIN}|$DOMAIN|g" "$TEMPLATE" > "$LIVE_CONF.new"
mv "$LIVE_CONF.new" "$LIVE_CONF"

# --- 5. Validate, then reload (roll back on failure) -----------------------
log "Validating the nginx configuration..."
if ! $COMPOSE exec -T nginx nginx -t; then
    log "Validation FAILED. Restoring the previous HTTP-only configuration."
    cp "$BACKUP_CONF" "$LIVE_CONF"
    $COMPOSE exec -T nginx nginx -s reload || true
    die "nginx rejected the TLS configuration; nothing was changed."
fi

log "Reloading nginx..."
$COMPOSE up -d nginx
$COMPOSE exec -T nginx nginx -s reload

# --- 6. Verify --------------------------------------------------------------
log "Verifying HTTPS..."
sleep 3
if curl -sfI "https://$DOMAIN/health" >/dev/null 2>&1; then
    log "HTTPS is serving on https://$DOMAIN"
elif [ -n "$STAGING_FLAG" ]; then
    log "HTTPS is up, but the staging certificate is untrusted (expected)."
else
    log "WARNING: could not verify https://$DOMAIN from this host."
    log "         This is often just DNS or an outbound firewall; check from outside."
fi

REDIRECT=$(curl -s -o /dev/null -w '%{http_code}' "http://$DOMAIN/" || echo "000")
log "HTTP -> HTTPS redirect returns: $REDIRECT (expect 301)"

cat <<EOF

Done. Next steps:
  * Renewal runs automatically: the 'certbot' service retries every 12h and
    nginx reloads every 6h to pick up a renewed certificate.
  * Verify the chain from outside: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN
  * HSTS starts at max-age=86400 (1 day). Once you are confident the
    certificate renews cleanly, raise it in
    $NGINX_DIR/tls-available/ml-monitor-tls.conf.template and re-run this script.
EOF
