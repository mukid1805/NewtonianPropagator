"""Generate API reference documentation pages and navigation dynamically."""

import ast
from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

CATEGORIES = {
    "Foundations": ["constants", "time", "ephemeris"],
    "Dynamics & Propagation": ["forces", "integrators", "propagator", "targeting"],
    "Mission Design": ["lambert", "flyby", "cr3bp"],
    "Constellations & Launch": ["swarm", "launchers"],
}

MODULE_TO_CATEGORY = {
    mod: cat for cat, modules in CATEGORIES.items() for mod in modules
}

core_dir = Path("core")


def get_public_functions_and_classes(file_path: Path):
    """Extract public top-level functions and classes with their constructor methods using AST."""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    members = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                members.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                members.append(node.name)
                # Check for explicit __init__ method inside the class to ensure constructor tables render
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef) and class_node.name == "__init__":
                        members.append(f"{node.name}.__init__")
    return members


for py_file in sorted(core_dir.glob("*.py")):
    module_name = py_file.stem
    if module_name.startswith("_"):
        continue

    category = MODULE_TO_CATEGORY.get(module_name, "Other Modules")
    cat_slug = category.lower().replace(" & ", "_").replace(" ", "_")

    doc_path = Path("reference", cat_slug, f"{module_name}.md")
    rel_path_from_ref = f"{cat_slug}/{module_name}.md"

    nav[category, module_name] = rel_path_from_ref

    public_members = get_public_functions_and_classes(py_file)

    with mkdocs_gen_files.open(doc_path, "w", encoding="utf-8") as fd:
        fd.write(
            f"# {module_name}\n\n"
            f"::: core.{module_name}\n"
            f"    options:\n"
            f"      show_root_heading: true\n"
            f"      show_source: true\n"
            f"      docstring_section_style: table\n"
        )

        if module_name == "constants" or not public_members:
            fd.write("      members: false\n")
        else:
            fd.write("      members:\n")
            for member in public_members:
                fd.write(f"        - {member}\n")

    mkdocs_gen_files.set_edit_path(doc_path, py_file)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w", encoding="utf-8") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
