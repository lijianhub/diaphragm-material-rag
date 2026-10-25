# The Node.js backend-for-frontend: compiled with tsc in a build stage; the runtime image
# has production dependencies only and runs as the unprivileged "node" user.
FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY bff/package.json bff/package-lock.json ./
RUN npm ci
COPY bff/tsconfig.json ./
COPY bff/src ./src
RUN npm run build

# Alpine for the runtime (261 MB instead of 348 MB on Debian slim): the app and its
# dependencies are 16 MB; the rest is the Node runtime itself (the binary alone is 124 MB).
FROM node:22-alpine
ENV NODE_ENV=production \
    BFF_HOST=0.0.0.0 \
    BFF_PORT=3000
WORKDIR /app
COPY bff/package.json bff/package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=build /app/dist ./dist
USER node
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD node -e "fetch('http://127.0.0.1:3000/api/health').then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))"
CMD ["node", "dist/index.js"]
