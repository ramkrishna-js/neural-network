import os
import sys
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import NoteProcessor, Tokenizer
from src.layers import RNN
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

def train_on_notes(file_path, epochs=10000):
    rprint(f"[bold cyan]Reading notes from {file_path}...[/bold cyan]")
    text = NoteProcessor.extract_text(file_path)
    if not text or len(text) < 10:
        rprint("[bold red]Not enough text found in file![/bold red]")
        return

    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    hidden_size = 100
    seq_length = 25
    
    rnn = RNN(vocab_size, hidden_size)
    
    rprint(f"[bold green]Training model on {len(text)} characters...[/bold green]")
    
    n, p = 0, 0
    hprev = np.zeros((hidden_size, 1))
    
    try:
        for epoch in range(epochs):
            if p + seq_length + 1 >= len(text) or epoch == 0:
                hprev = np.zeros((hidden_size, 1))
                p = 0
            
            inputs = tokenizer.encode(text[p : p+seq_length])
            targets = tokenizer.encode(text[p+1 : p+seq_length+1])
            
            xs, hs, ps = rnn.forward(inputs, hprev)
            hprev = rnn.backward(xs, hs, ps, targets, 0.1)
            
            if epoch % 1000 == 0:
                rprint(f"[dim]Epoch {epoch}, Loss: {-np.sum(np.log(ps[0][targets[0]])):.4f}[/dim]")
                sample_text = rnn.sample(hprev, inputs[0], 50, tokenizer)
                rprint(Panel(sample_text, title=f"Sample at Epoch {epoch}"))
            
            p += seq_length
            n += 1
            
        # Save model
        model_data = {'rnn': rnn, 'tokenizer': tokenizer}
        with open('models/notes_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        rprint("[bold green]Model trained and saved to models/notes_model.pkl[/bold green]")
        
    except KeyboardInterrupt:
        rprint("[bold yellow]Training interrupted. Saving current state...[/bold yellow]")
        model_data = {'rnn': rnn, 'tokenizer': tokenizer}
        with open('models/notes_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

def chat():
    if not os.path.exists('models/notes_model.pkl'):
        rprint("[bold red]No trained notes model found! Run training first.[/bold red]")
        return

    with open('models/notes_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    rnn = data['rnn']
    tokenizer = data['tokenizer']
    hidden_size = rnn.hidden_size
    
    rprint(Panel("[bold green]Study Buddy AI[/bold green]\nI've learned from your notes. Ask me anything or type 'exit' to quit.", border_style="blue"))
    
    h = np.zeros((hidden_size, 1))
    
    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # We'll use the last char of user input as seed
        seed_ix = tokenizer.encode(user_input)[-1] if tokenizer.encode(user_input) else 0
        
        # Generate a response
        response = rnn.sample(h, seed_ix, 100, tokenizer)
        
        rprint(f"[bold magenta]Study Buddy:[/bold magenta] {response}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    elif len(sys.argv) > 2 and sys.argv[1] == "train":
        train_on_notes(sys.argv[2])
    else:
        rprint("[bold yellow]Usage:[/bold yellow]")
        rprint("  python scripts/study_buddy.py train <path_to_notes>")
        rprint("  python scripts/study_buddy.py chat")
