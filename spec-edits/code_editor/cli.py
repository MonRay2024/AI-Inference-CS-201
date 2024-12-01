import click
from pathlib import Path
from code_editor.core.code_editor import CodeEditor
from code_editor.utils.model_loader import load_model
from code_editor.utils.config import EditorConfig
from code_editor.utils.logger import EditorLogger

@click.group()
def cli():
    """Code Editor CLI"""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('instruction')
@click.option('--config', '-c', type=click.Path(), help='Path to config file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def edit(input_file: str, instruction: str, config: Optional[str], output: Optional[str]):
    """Edit code according to instruction"""
    logger = EditorLogger()
    
    # Load configuration
    if config:
        config_obj = EditorConfig.from_dict(yaml.safe_load(Path(config).read_text()))
    else:
        config_obj = EditorConfig()
    
    # Initialize editor
    model, tokenizer = load_model(config_obj.model_name)
    editor = CodeEditor(model, tokenizer, config_obj)
    
    # Read input file
    code = Path(input_file).read_text()
    
    # Process edit
    edited_code, operations, impact_scores = editor.perfect_edit(code, instruction)
    
    # Write output
    if output:
        Path(output).write_text(edited_code)
    else:
        click.echo(edited_code)

if __name__ == '__main__':
    cli() 