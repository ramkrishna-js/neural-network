import os
import sys
from rich.tree import Tree
from rich.console import Console

def generate_tree():
    console = Console()
    # Go up one level to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    project_name = os.path.basename(os.getcwd())
    tree = Tree(f"[bold green]{project_name}[/bold green] 🧠")

    ignore = {'.git', 'venv', '__pycache__', 'data'}

    def add_to_tree(path, branch):
        for item in sorted(os.listdir(path)):
            if item in ignore: continue
            
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                node = branch.add(f"[bold blue]{item}/[/bold blue]")
                add_to_tree(full_path, node)
            else:
                if item.endswith('.py'):
                    branch.add(f"[green]{item}[/green] 🐍")
                elif item.endswith('.pkl'):
                    branch.add(f"[magenta]{item}[/magenta] 🤖")
                elif item == 'README.md':
                    branch.add(f"[cyan]{item}[/cyan] 📝")
                elif item == 'LICENSE':
                    branch.add(f"[yellow]{item}[/yellow] 📜")
                else:
                    branch.add(f"[white]{item}[/white]")

    add_to_tree(".", tree)
    console.print(tree)

if __name__ == "__main__":
    generate_tree()