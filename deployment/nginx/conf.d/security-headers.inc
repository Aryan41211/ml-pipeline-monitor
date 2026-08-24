# =============================================================================
# Shared security response headers.
# =============================================================================
# Named .inc, not .conf, so nginx.conf's `include conf.d/*.conf` does not try
# to parse it as a standalone config.
#
# WHY THIS FILE EXISTS: nginx's add_header directives are inherited from an
# outer level ONLY IF the current level declares none of its own. Declaring a
# single add_header inside a server or location therefore silently discards
# every header set in http{}. Include this file at every level that adds any
# header of its own, so the set is always complete.
#
# HSTS is deliberately NOT here: it is meaningless over plain HTTP and belongs
# only to the TLS server, which adds it alongside this include.
# =============================================================================

add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' wss: https:;" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
