# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Contact**: security@agenticraft.ai

You may also use [GitHub Security Advisories](https://github.com/agenticraft/agenticraft-foundation/security/advisories) to report vulnerabilities privately.

Do NOT open a public GitHub issue for security vulnerabilities.

## Severity Tiers

| Severity | Response SLA | Examples |
|----------|-------------|----------|
| Critical | 24 hours | Remote code execution, deserialization attacks, secret exposure |
| High | 72 hours | Privilege escalation, injection attacks |
| Medium | 1 week | Information disclosure, denial of service |
| Low | Best effort | Hardening recommendations, minor issues |

## Scope

The following are in scope:

- All code in the `agenticraft-foundation` package
- Verification and model-checking components
- Serialization and deserialization of specifications
- Dependency vulnerabilities

The following are out of scope:

- Denial of service attacks against infrastructure
- Social engineering
- Issues in third-party dependencies (report upstream, notify us)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x | Yes |

## Disclosure Process

1. Report via email (security@agenticraft.ai) or GitHub Security Advisory
2. Include: description, reproduction steps, impact assessment
3. We will acknowledge receipt within the response SLA for the severity tier
4. We will work with you to understand and resolve the issue
5. Once fixed, we will coordinate disclosure timing with you
