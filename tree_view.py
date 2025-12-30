import os
from rich.tree import Tree
from rich.console import Console
from rich import print

def generate_tree():
    console = Console()
    project_name = os.path.basename(os.getcwd())
    tree = Tree(f"[bold green]{project_name}[/bold green] 🧠")

    # Files to ignore
    ignore = {'.git', 'venv', '__pycache__', 'data'}

    for root, dirs, files in os.walk("."):
        # Filter directories and files
        dirs[:] = [d for d in dirs if d not in ignore]
        
        # Calculate depth for tree branching
        level = root.replace(".", "").count(os.sep)
        
        # We only care about the top level for this simple view
        if root == ".":
            for d in sorted(dirs):
                tree.add(f"[bold blue]{d}/[/bold blue]")
            for f in sorted(files):
                if f.endswith('.py'):
                    tree.add(f"[green]{f}[/green] 🐍")
                elif f == 'README.md':
                    tree.add(f"[cyan]{f}[/cyan] 📝")
                elif f == 'LICENSE':
                    tree.add(f"[yellow]{f}[/yellow] 📜")
                elif f == 'model.pkl':
                    tree.add(f"[magenta]{f}[/magenta] 🤖")
                else:
                    tree.add(f"[white]{f}[/white]")

    console.print("\n[bold green]Project Structure View:[/bold green]")
    console.print(tree)
    console.print("\n")

if __name__ == "__main__":
    generate_tree()
