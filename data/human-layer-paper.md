<!-- DRAFT: based on the project brief and the citation on https://aidxlab.github.io/. The Springer abstract could not be retrieved; please replace this summary with the official abstract. -->

# The Human Layer of Agentic AI Memory

## Citation
Parameswaran, A., Hussain, W., & Hossain, M. N. (2026). The Human Layer of Agentic
AI Memory: What Self-Improving LLM Systems Actually Learn in Production. In
AI-Driven Mathematics Education (pp. 35–52). Springer, Cham.

## Main finding
The chapter studies what self-improving LLM systems actually store and reuse in
their memory once deployed in production. It found that the memory items reused
most are user-specific "intent-calibration" rules rather than domain knowledge,
at a ratio of roughly 3.4 to 1.

## What is an intent-calibration rule?
An intent-calibration rule records how a particular user communicates, what they
mean when their wording is ambiguous, and how they want responses delivered. For
example: "When this user says 'quick summary', they want three bullet points."

## What is domain knowledge memory?
Domain knowledge memory stores facts about the subject area, such as details of a
course, a policy or a dataset. These facts are usually shared by many users.

## Why it matters
The finding suggests that agent memory should be designed around the "human
layer": learning each person's intent and preferences, with transparency and user
control, rather than only accumulating more facts. The AIDX Assistant prototype
demonstrates and measures this distinction.
