
import os
FILES = ["README.md", "REPORT.md", "DECISIONS.md", "requirements.txt", ".env.example", ".gitignore"]
DIRS = [
    "data/raw", "data/processed", "data/golden", "data/sample",
    "src", "src/data", "src/intents", "src/retrieval", "src/generation",
    "src/escalation", "src/baselines", "src/evaluation",
    "scripts", "tests", "results",
]
INIT_PY_DIRS = [
    "src", "src/data", "src/intents", "src/retrieval", "src/generation",
    "src/escalation", "src/baselines", "src/evaluation",
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)
for d in INIT_PY_DIRS:
    open(os.path.join(d, "__init__.py"), "a").close()
for f in FILES:
    open(f, "a").close()
open(os.path.join("results", ".gitkeep"), "a").close()

print("Project structure created.")