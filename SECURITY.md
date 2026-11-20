# Security Policy

Thank you for helping keep this project and its users safe.

This security policy applies to this repository and can be reused across the author's public projects.

## Supported Versions

Security fixes are generally applied to the default branch and the latest actively maintained release.

| Version / Branch | Supported |
| --- | --- |
| `main` / default branch | Yes |
| Latest stable release | Yes, when applicable |
| Older releases | No, unless explicitly stated in the project documentation |
| Development or archived branches | No |

If this repository uses a different release policy, the project README or release notes take precedence.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub Issues, pull requests, discussions, or social media.

Use one of the following private channels:

1. **GitHub Private Vulnerability Reporting**: open the repository's **Security** tab and choose **Report a vulnerability**, when this feature is enabled.

1. **Private security contact**: contact the repository owner privately through GitHub if private vulnerability reporting is unavailable.

When reporting a vulnerability, please include:

- A clear description of the issue and its potential impact.

- The affected repository, branch, tag, version, or commit.

- Reproduction steps or a minimal proof of concept.

- Any required configuration, input, dataset, or environment details.

- Suggested remediation, if you have one.

- Your preferred contact details for follow-up.

Please avoid including real secrets, passwords, API keys, personal data, or sensitive production data in the report. Redact them from logs and examples.

## Response Process

The maintainer will make a reasonable effort to:

- Acknowledge receipt of a private report within 7 calendar days.

- Investigate and assess the report as soon as practical.

- Keep the reporter informed when additional information is required.

- Coordinate a fix, mitigation, or documented workaround when appropriate.

- Credit the reporter in the release notes or security advisory when the reporter agrees.

Response times may vary depending on the severity, reproducibility, project activity, and availability of a safe fix.

## Disclosure Policy

Please allow reasonable time for investigation and remediation before publicly disclosing a vulnerability.

The maintainer may publish a security advisory, release note, or changelog entry after a fix or mitigation is available. Disclosure timing and content will be coordinated with the reporter whenever practical.

## Scope

Examples of issues that may be considered security vulnerabilities include:

- Authentication or authorization bypasses.

- Exposure of credentials, tokens, private data, or personal information.

- Remote code execution, command injection, or unsafe deserialization.

- Server-side request forgery, path traversal, or arbitrary file access.

- Cross-site scripting, SQL injection, or other injection vulnerabilities.

- Insecure handling of uploaded files, model files, datasets, or user-controlled input.

- Vulnerable default configuration that creates a meaningful security risk.

Issues that are normally outside the security-reporting scope include ordinary bugs without a security impact, unsupported environments, rate-limit complaints, feature requests, and vulnerabilities in third-party services that are not caused by this project.

## Safe Harbor

Security research performed in good faith is welcome. Please avoid actions that could harm users, degrade availability, access data that does not belong to you, or violate applicable laws.

Do not perform denial-of-service testing, social engineering, spam, destructive testing, or testing against other users' accounts or production data. Stop testing and report the issue privately once you have enough evidence to demonstrate the problem.

We will not pursue legal action against researchers who follow this policy in good faith and avoid the activities described above.

## Dependency and Secret Hygiene

Contributors should:

- Never commit passwords, API keys, tokens, private certificates, or other secrets.

- Use environment variables or a secret manager for sensitive configuration.

- Keep dependencies and base images reasonably up to date.

- Review automated dependency and code-scanning alerts.

- Remove personal or production data from examples, notebooks, logs, and test fixtures.

For project-specific security instructions, the repository README and deployment documentation may provide additional requirements.
