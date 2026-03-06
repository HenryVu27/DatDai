# Auth Design: Individual User Accounts

## Overview
Simple authentication system for the DatDai chatbot. Individual username/password accounts managed by an admin, with secure session cookies.

## Database Schema

### `users` table
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | Primary key |
| username | TEXT | Unique, not null |
| password_hash | TEXT | bcrypt hash |
| is_admin | BOOLEAN | Default false |
| created_at | TEXT | ISO timestamp |

### `auth_sessions` table
| Column | Type | Notes |
|--------|------|-------|
| token | TEXT | Primary key (random 32-byte hex) |
| user_id | TEXT | FK to users.id |
| expires_at | TEXT | ISO timestamp, 7 days from creation |

## Login Flow
1. User visits any page -> middleware checks session cookie
2. No valid session -> redirect to `/login`
3. User submits username/password to `POST /auth/login`
4. Server verifies bcrypt hash, creates auth_session row, sets cookie
5. Cookie flags: `HttpOnly`, `Secure`, `SameSite=Strict`
6. Subsequent requests validated via cookie -> auth_sessions lookup

## Auth Middleware
- FastAPI dependency applied to all routes
- Exempt routes: `/login`, `/auth/login`, `/health`, `/static/*`
- Invalid/expired session -> 401 + redirect to `/login`

## Admin
- `/admin` page accessible only to `is_admin=True` users
- Features: list users, add user (username + temp password), remove user
- First admin auto-created on startup from `ADMIN_PASSWORD` env var (username: `admin`)

## Chat Session Ownership
- `sessions` table gets `user_id` column
- Users only see their own chat history
- Existing sessions (no user_id) remain accessible to admin only

## Configuration
| Env var | Purpose | Required |
|---------|---------|----------|
| `ADMIN_PASSWORD` | Initial admin account password | First run only |
| `SESSION_SECRET` | Cookie signing key | Auto-generated if not set |

## Dependencies
- `bcrypt` -- password hashing
- No other new dependencies

## Security
- Passwords: bcrypt with default work factor
- Cookies: HttpOnly + Secure + SameSite=Strict
- Sessions: 7-day expiry, stored server-side
- Admin endpoints: is_admin check middleware
