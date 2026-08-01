# =============================================================================
# Frontend image - build the SPA with Node, serve the static output with nginx.
#
# The runtime stage carries no Node and no node_modules: what ships is the
# contents of dist/ plus nginx (~50MB vs ~1.5GB for a node image with deps).
#
# Build context is frontend/ (see docker-compose.yml):
#   docker build -f frontend/Dockerfile -t retail-insights-web frontend
# =============================================================================

# -------------------------------------------------------------------- build --
FROM node:20-alpine AS build

WORKDIR /app

# package.json + lockfile first, so the dependency layer is cached against the
# lockfile hash rather than being invalidated by every source edit.
COPY package.json package-lock.json ./

# `npm ci` (not `install`) installs exactly what the lockfile pins and fails if
# the two have drifted - the behaviour you want in an image build, where a
# silently upgraded transitive dep is a reproducibility bug.
RUN npm ci --no-audit --no-fund

COPY . .

# Vite substitutes VITE_* variables at BUILD time and inlines them into the
# bundle; there is no process.env to read at runtime. So the API URL has to
# arrive as a build arg, and pointing the image at a different backend means
# rebuilding it. The default suits docker-compose, where the browser reaches
# the published backend port on the host.
#
# Nothing secret goes here: anything inlined into a bundle is readable by
# anyone who loads the page. Only the public API base URL belongs in this arg.
ARG VITE_API_BASE_URL=http://localhost:8000/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Runs tsc --noEmit then vite build (see package.json), so a type error fails
# the image build instead of shipping a broken bundle.
RUN npm run build

# ------------------------------------------------------------------ runtime --
FROM nginx:1.27-alpine AS runtime

# Replace the stock server block with the SPA config (history fallback,
# asset caching, /healthz).
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/healthz || exit 1

# nginx defaults to daemonising, which would exit PID 1 immediately and stop
# the container; "daemon off" keeps it in the foreground.
CMD ["nginx", "-g", "daemon off;"]
