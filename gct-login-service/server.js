const express = require('express');
const session = require('express-session');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const AppleStrategy = require('passport-apple');
const fs = require('fs');
const path = require('path');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 4000;

// log directory
const logDir = path.join(__dirname, 'logs');
fs.mkdirSync(logDir, { recursive: true });
const accessLogStream = fs.createWriteStream(path.join(logDir, 'access.log'), { flags: 'a' });
const errorLogStream = fs.createWriteStream(path.join(logDir, 'error.log'), { flags: 'a' });

app.use(morgan('combined', { stream: accessLogStream }));
app.use(express.urlencoded({ extended: false }));
app.use(session({ secret: 'gct-secret', resave: false, saveUninitialized: false }));
app.use(passport.initialize());
app.use(passport.session());

passport.serializeUser((user, done) => done(null, user));
passport.deserializeUser((obj, done) => done(null, obj));

if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: '/auth/google/callback',
  }, (accessToken, refreshToken, profile, cb) => cb(null, profile)));
}

if (process.env.APPLE_CLIENT_ID && process.env.APPLE_TEAM_ID && process.env.APPLE_KEY_ID && process.env.APPLE_PRIVATE_KEY) {
  passport.use(new AppleStrategy({
    clientID: process.env.APPLE_CLIENT_ID,
    teamID: process.env.APPLE_TEAM_ID,
    keyID: process.env.APPLE_KEY_ID,
    privateKeyString: process.env.APPLE_PRIVATE_KEY,
    callbackURL: '/auth/apple/callback',
  }, (accessToken, refreshToken, idToken, profile, cb) => cb(null, profile)));
}

function ensureAuth(req, res, next) {
  if (req.isAuthenticated()) { return next(); }
  res.redirect('/login');
}

app.get('/login', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Login</title></head>
<body>
  <h1>Login</h1>
  <a href="/auth/google">Login with Google</a><br/>
  <a href="/auth/apple">Login with Apple</a>
</body>
</html>`);
});

app.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));
app.get('/auth/google/callback', passport.authenticate('google', { failureRedirect: '/login' }), (req, res) => {
  res.redirect('/protected');
});

app.get('/auth/apple', passport.authenticate('apple'));
app.post('/auth/apple/callback', passport.authenticate('apple', { failureRedirect: '/login' }), (req, res) => {
  res.redirect('/protected');
});

app.get('/protected', ensureAuth, (req, res) => {
  res.send(`Hello ${req.user.displayName || 'User'}! <a href="/logout">Logout</a>`);
});

app.get('/logout', (req, res) => {
  req.logout(() => {
    res.redirect('/login');
  });
});

app.use((err, req, res, next) => {
  errorLogStream.write(`${new Date().toISOString()} - ${err.stack}\n`);
  res.status(500).send('Internal Server Error');
});

app.listen(PORT, () => {
  console.log(`Login service running on port ${PORT}`);
});
