# Security Policy

## Scope

This is a local, notebook-based research/portfolio project — there is no deployed service, no API, and no user-facing endpoint. The realistic security surface is small:

- **Dependency vulnerabilities** in packages the notebook imports (TensorFlow/PyTorch, NumPy, etc.)
- **Unsafe deserialization** if the notebook loads pickled model weights or data from an untrusted source
- **Notebook execution risk** — as with any `.ipynb`, don't run cells from a copy of this repo you don't trust, since notebook cells execute arbitrary code

There is no authentication, no stored user data, and no network-facing component to report vulnerabilities against beyond the above.

## Supported Versions

Only the latest commit on `main` is supported.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Reporting a Vulnerability

For dependency vulnerabilities or anything else in scope above, please report via:
- GitHub's [private vulnerability reporting](https://github.com/Ritik650/Image-Captioning-using-Deep-Learning/security/advisories/new) (Security tab → Report a vulnerability)
- Email: ry9812262@gmail.com

This is a solo-maintained project, so response times aren't guaranteed, but reports will be acknowledged and addressed as soon as possible.
