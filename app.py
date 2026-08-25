import gradio as gr

from src.examples import CODE_EXAMPLES, ERROR_EXAMPLES, DOCKER_EXAMPLES
from src.model_engine import process_code, process_error, process_env

# Create dictionaries for dropdowns
code_examples_dict = {item[0]: item[1] for item in CODE_EXAMPLES}
error_examples_dict = {item[0]: item[1] for item in ERROR_EXAMPLES}
env_examples_dict = {item[0]: item[1] for item in DOCKER_EXAMPLES}

code_example_keys = list(code_examples_dict.keys())
error_example_keys = list(error_examples_dict.keys())
env_example_keys = list(env_examples_dict.keys())

css = """
body {
    background-color: #0f172a;
    color: #e2e8f0;
}
.container {
    max-width: 900px !important;
    margin: 0 auto;
    padding-top: 2rem;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.85rem;
    color: #64748b;
    padding-bottom: 2rem;
}
.block-card {
    border: 1px solid #334155;
    border-radius: 8px;
    background-color: #1e293b;
    padding: 1rem;
    min-height: 100px;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("<div class='title'>ROCmPilot</div>")
            gr.Markdown("<div class='subtitle'>Fine-tuned code migration assistant for AMD ROCm.</div>")
            
        with gr.Tabs():
            # Tab 1: Code Migration
            with gr.Tab("Code Migration"):
                gr.Markdown("Identify hardcoded NVIDIA assumptions in Python scripts.")
                
                with gr.Row():
                    code_dropdown = gr.Dropdown(choices=code_example_keys, label="Load example", value=None)
                
                with gr.Row():
                    code_input = gr.Code(language="python", lines=12, label="Input Code")
                
                with gr.Row():
                    analyze_code_btn = gr.Button("Analyze code", variant="primary")
                    
                with gr.Row():
                    code_output = gr.Markdown(elem_classes="block-card", label="Analysis Result")
                
                def load_code_example(key):
                    return code_examples_dict.get(key, "")
                    
                code_dropdown.change(fn=load_code_example, inputs=code_dropdown, outputs=code_input)
                analyze_code_btn.click(fn=process_code, inputs=code_input, outputs=code_output)

            # Tab 2: Error Explainer
            with gr.Tab("Error Explainer"):
                gr.Markdown("Get actionable debugging steps for runtime error logs.")
                
                with gr.Row():
                    error_dropdown = gr.Dropdown(choices=error_example_keys, label="Load example", value=None)
                
                with gr.Row():
                    error_input = gr.Code(language="markdown", lines=12, label="Error Log")
                
                with gr.Row():
                    analyze_error_btn = gr.Button("Explain error", variant="primary")
                    
                with gr.Row():
                    error_output = gr.Markdown(elem_classes="block-card", label="Analysis Result")
                
                def load_error_example(key):
                    return error_examples_dict.get(key, "")
                    
                error_dropdown.change(fn=load_error_example, inputs=error_dropdown, outputs=error_input)
                analyze_error_btn.click(fn=process_error, inputs=error_input, outputs=error_output)

            # Tab 3: Environment Fixer
            with gr.Tab("Environment Fixer"):
                gr.Markdown("Find unportable base images and NVIDIA dependencies.")
                
                with gr.Row():
                    env_dropdown = gr.Dropdown(choices=env_example_keys, label="Load example", value=None)
                
                with gr.Row():
                    env_input = gr.Code(language="dockerfile", lines=12, label="Dockerfile / requirements.txt")
                
                with gr.Row():
                    analyze_env_btn = gr.Button("Fix environment", variant="primary")
                    
                with gr.Row():
                    env_output = gr.Markdown(elem_classes="block-card", label="Analysis Result")
                
                def load_env_example(key):
                    return env_examples_dict.get(key, "")
                    
                env_dropdown.change(fn=load_env_example, inputs=env_dropdown, outputs=env_input)
                analyze_env_btn.click(fn=process_env, inputs=env_input, outputs=env_output)

        with gr.Column(elem_classes="footer"):
            gr.Markdown("ROCmPilot provides migration guidance. Always test generated changes in your target ROCm environment.")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=css,
        theme=gr.themes.Monochrome(),
    )
