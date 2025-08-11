"""Content extraction utilities for different file types."""

from pathlib import Path
from typing import Optional


def extract_code_content(file_path: str, content: str) -> str:
    """Extract smart content from code files based on file type and size."""
    file_size = len(content)

    # For small files (<4KB), embed full content
    if file_size < 4096:
        return format_code_content(file_path, content, full_content=True)

    # For larger files, extract metadata
    if file_path.endswith(".py"):
        return extract_python_metadata(file_path, content)
    elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
        return extract_typescript_metadata(file_path, content)
    else:
        # Fallback: first part + summary
        lines = content.split("\n")
        return format_code_content(
            file_path,
            "\n".join(lines[:50]) + f"\n\n# ... (truncated {len(lines)-50} lines)",
            full_content=False,
        )


def format_code_content(file_path: str, content: str, full_content: bool) -> str:
    """Format code content for embedding with metadata."""
    filename = Path(file_path).name

    header = f"""# Code: {file_path}
**File**: {filename}
**Type**: {"Full Content" if full_content else "Smart Extract"}
**Size**: {len(content)} characters

## Implementation

```{"python" if file_path.endswith(".py") else "typescript"}
{content}
```"""

    return header


def extract_python_metadata(file_path: str, content: str) -> str:
    """Extract metadata from Python files."""
    lines = content.split("\n")

    # Extract imports, classes, functions, and docstrings
    imports = []
    classes = []
    functions = []
    current_docstring = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Imports
        if line.startswith(("import ", "from ")):
            imports.append(line)

        # Classes
        elif line.startswith("class "):
            class_def = line
            if ":" not in line and i + 1 < len(lines):
                class_def += " " + lines[i + 1].strip()
            classes.append(class_def)

        # Functions and methods
        elif line.startswith("def ") or "    def " in line:
            func_def = line.strip()
            if ":" not in line and i + 1 < len(lines):
                func_def += " " + lines[i + 1].strip()
            functions.append(func_def)

        # Module docstring
        elif i < 10 and line.startswith('"""') and current_docstring is None:
            docstring_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith('"""'):
                docstring_lines.append(lines[i])
                i += 1
            if i < len(lines):
                docstring_lines.append(lines[i])
            current_docstring = "\n".join(docstring_lines)

        i += 1

    # Format extracted content
    extracted = f"""# Code: {file_path}
**Type**: Python Smart Extract
**Classes**: {len(classes)}
**Functions**: {len(functions)}
**Imports**: {len(imports)}

## Module Documentation
{current_docstring or "No module docstring found"}

## Imports
```python
{chr(10).join(imports[:10])}
{f"... and {len(imports)-10} more imports" if len(imports) > 10 else ""}
```

## Classes
```python
{chr(10).join(classes)}
```

## Functions
```python
{chr(10).join(functions[:15])}
{f"... and {len(functions)-15} more functions" if len(functions) > 15 else ""}
```"""

    return extracted


def extract_typescript_metadata(file_path: str, content: str) -> str:
    """Extract metadata from TypeScript/JavaScript files."""
    # Limit scan to first 4000 lines – avoids excessive work on megafile bundles
    lines = content.split("\n")[:4000]

    # Extract imports, interfaces, types, components, functions
    imports = []
    interfaces = []
    types = []
    components = []
    functions = []
    exports = []

    for line in lines:
        stripped = line.strip()

        # Imports
        if stripped.startswith("import "):
            imports.append(stripped)

        # Interfaces
        elif stripped.startswith("interface "):
            interfaces.append(stripped)

        # Types
        elif stripped.startswith("type "):
            types.append(stripped)

        # React components (function components)
        elif "const " in stripped and (
            "React.FC" in stripped or "JSX.Element" in stripped
        ):
            components.append(stripped)
        elif stripped.startswith("function ") and (
            "React" in content or "JSX" in content
        ):
            components.append(stripped)

        # Functions
        elif (
            stripped.startswith("function ")
            or stripped.startswith("const ")
            and "=>" in stripped
        ):
            functions.append(stripped)

        # Exports
        elif stripped.startswith("export "):
            exports.append(stripped)

    # If lists are huge, truncate to keep extract compact
    def _truncate(lst, max_items):
        return lst[:max_items], len(lst) - max_items if len(lst) > max_items else 0

    imports_head, imports_more = _truncate(imports, 8)
    interfaces_head, interfaces_more = _truncate(interfaces, 8)
    types_head, types_more = _truncate(types, 8)
    components_head, components_more = _truncate(components, 10)
    functions_head, functions_more = _truncate(functions, 10)

    # Format extracted content
    extracted = f"""# Code: {file_path}
**Type**: TypeScript/JavaScript Smart Extract
**Imports**: {len(imports)}
**Interfaces**: {len(interfaces)}
**Types**: {len(types)}
**Components**: {len(components)}
**Functions**: {len(functions)}

## Imports
```typescript
{chr(10).join(imports_head)}
{f"... and {imports_more} more imports" if imports_more else ""}
```

## Interfaces & Types
```typescript
{chr(10).join(interfaces_head + types_head)}
{f"... and {interfaces_more + types_more} more" if interfaces_more + types_more else ""}
```

## Components
```typescript
{chr(10).join(components_head)}
{f"... and {components_more} more components" if components_more else ""}
```

## Functions
```typescript
{chr(10).join(functions_head)}
{f"... and {functions_more} more functions" if functions_more else ""}
```

## Exports
```typescript
{chr(10).join(exports)}
```"""

    return extracted


def extract_title_from_content(content: str, file_path: str = "") -> Optional[str]:
    """Extract title from file content, with fallback to filename."""
    lines = content.split("\n")

    # Look for first header
    for line in lines[:15]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        elif line.startswith("## "):
            return line[3:].strip()

    # Look for title in YAML frontmatter
    if content.startswith("---"):
        try:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip()
                        return title.strip("\"'")
        except Exception:
            pass

    # Fallback to filename
    if file_path:
        return Path(file_path).stem.replace("_", " ").replace("-", " ").title()

    return "Untitled"
