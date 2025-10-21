🧙‍♂️ Alt-Text Wizard
Generate accessible, descriptive alt text for images using OpenAI's GPT-4o Vision API—with smart device name substitution, cost tracking, and bulk processing support.

✨ Features
Single or bulk image uploads
Automatic alt text generation optimized for screen readers
Override generic terms (e.g., "vape pen" → "VEEV ONE") with a custom device name
Real-time cost estimation based on OpenAI’s token pricing
Session statistics: API calls, execution time, token usage, and total cost
Optimized bulk mode: Process multiple images in a single API call to reduce cost
Export results as CSV (bulk mode)
▶️ How to Run
Install dependencies:


pip install -r requirements.txt
Get your OpenAI API key from https://platform.openai.com/api-keys

Run the app:


streamlit run alt-tag-generator.py
Enter your API key in the sidebar and start generating alt text!

Your API key is stored only in memory and never saved or transmitted beyond the OpenAI API.

📦 Requirements
See requirements.txt for dependencies.

Made with ❤️ and OpenAI GPT-4o Vision.
