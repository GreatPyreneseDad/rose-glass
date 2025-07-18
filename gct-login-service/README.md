# GCT Login Service

A simple Node.js Express server providing a login page with Google and Apple authentication links. Crash logs are written to `logs/error.log` and HTTP access is logged to `logs/access.log`.

## Setup
1. Run `npm install` in this directory.
2. Set environment variables for OAuth credentials:
   - `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
   - `APPLE_CLIENT_ID`, `APPLE_TEAM_ID`, `APPLE_KEY_ID`, and `APPLE_PRIVATE_KEY`
3. Start the server:

```bash
node server.js
```

The login page will be available at `http://localhost:4000/login`.
