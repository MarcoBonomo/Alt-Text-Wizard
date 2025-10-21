import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import time
from datetime import datetime
import re

# Set page config
st.set_page_config(
    page_title="Alt-Text Wizard",
    page_icon="🧙‍♂️",
    layout="wide"
)

# Initialize session state
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = {'input': 0, 'output': 0}
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'total_execution_time' not in st.session_state:
    st.session_state.total_execution_time = 0.0
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = None

# Title and description
st.title("Alt-Text Wizard 🧙‍♂️✨")
st.markdown("**Get instant, accessible alt text for your images—powered by OpenAI's Vision API and a sprinkle of automation magic.**")

# Pricing constants (as of 2024, verify current pricing)
PRICING = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,  # $2.50 per 1M input tokens
        "output": 10.00 / 1_000_000  # $10.00 per 1M output tokens
    }
}

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. It will only be stored for this session."
    )
    
    if api_key:
        st.success("API Key provided ✓")
    else:
        st.warning("Please enter your API Key to continue")
    
    st.markdown("---")
    
    # Cost tracking section
    st.header("📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Calls", st.session_state.api_calls)
    with col2:
        st.metric("⏱️ Total Time", f"{st.session_state.total_execution_time:.2f}s")
    
    st.metric("💰 Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Average metrics
    if st.session_state.api_calls > 0:
        avg_time = st.session_state.total_execution_time / st.session_state.api_calls
        avg_cost = st.session_state.total_cost / st.session_state.api_calls
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Avg Time", f"{avg_time:.2f}s")
        with col4:
            st.metric("Avg Cost", f"${avg_cost:.4f}")
    
    with st.expander("🔢 Token Details"):
        st.write(f"**Input tokens:** {st.session_state.total_tokens['input']:,}")
        st.write(f"**Output tokens:** {st.session_state.total_tokens['output']:,}")
        st.write(f"**Total tokens:** {st.session_state.total_tokens['input'] + st.session_state.total_tokens['output']:,}")
    
    if st.button("Reset Statistics", use_container_width=True):
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = {'input': 0, 'output': 0}
        st.session_state.api_calls = 0
        st.session_state.total_execution_time = 0.0
        st.session_state.current_result = None
        st.session_state.bulk_results = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses OpenAI's GPT-4o Vision model to generate descriptive alt text for images.
    
    **Features:**
    - Single image processing
    - Bulk image processing (optimized)
    - Download results as CSV
    - Cost estimation & tracking
    - Execution timing
    """)

# Function to encode image to base64
def encode_image(image_file):
    """Convert uploaded file to base64 string"""
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to calculate cost
def calculate_cost(usage, model="gpt-4o"):
    """Calculate cost based on token usage"""
    input_cost = usage.prompt_tokens * PRICING[model]["input"]
    output_cost = usage.completion_tokens * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Function to substitute device name in alt text
def substitute_device_name(alt_text, device_name):
    """
    Replace generic product terms or redundant device names with specific device name.
    Handles cases like:
    - "vape pen" -> device_name
    - "VEEV VEEV ONE" -> device_name
    - Duplicate brand names
    """
    if not device_name or not device_name.strip():
        return alt_text
    
    device_name = device_name.strip()
    
    # Step 1: Handle redundant/duplicate brand patterns (e.g., "VEEV VEEV ONE", "IQOS IQOS ILUMA")
    # Look for repeated words that might be brand duplications
    words = alt_text.split()
    cleaned_words = []
    i = 0
    while i < len(words):
        # Check if current word is repeated in the next 1-2 words
        if i < len(words) - 1 and words[i].upper() == words[i + 1].upper():
            # Skip the duplicate
            cleaned_words.append(words[i])
            i += 2
        else:
            cleaned_words.append(words[i])
            i += 1
    
    alt_text = ' '.join(cleaned_words)
    
    # Step 2: List of generic device terms to replace (in order of specificity)
    generic_patterns = [
        # Specific vaping terms
        (r'\b(vape\s+pen|vaping\s+device|vaporizer|e-cigarette|electronic\s+cigarette)\b', device_name),
        # Brand + model patterns (e.g., "VEEV ONE", "IQOS ILUMA")
        (r'\b([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z0-9]+)?)\b', device_name),
        # Single all-caps brand names (e.g., "VEEV", "IQOS")
        (r'\b([A-Z]{3,})\b', device_name),
        # Generic "device" as last resort
        (r'\bdevice\b', device_name),
    ]
    
    # Step 3: Try each pattern and stop after first successful replacement
    for pattern, replacement in generic_patterns:
        new_text = re.sub(pattern, replacement, alt_text, count=1, flags=re.IGNORECASE)
        if new_text != alt_text:
            # Successful replacement, clean up any double spaces
            new_text = re.sub(r'\s+', ' ', new_text).strip()
            return new_text
    
    # Step 4: If no pattern matched, try to find any product noun phrase
    # Look for common product descriptors followed by a potential brand/model
    product_phrase_pattern = r'\b((?:purple|green|blue|red|black|white|silver|gold)\s+)?([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\b'
    match = re.search(product_phrase_pattern, alt_text)
    
    if match:
        # Replace the matched phrase with color (if present) + device name
        color_prefix = match.group(1) if match.group(1) else ""
        replacement = f"{color_prefix}{device_name}"
        new_text = alt_text[:match.start()] + replacement + alt_text[match.end():]
        new_text = re.sub(r'\s+', ' ', new_text).strip()
        return new_text
    
    # If nothing matched, return original text
    return alt_text

# Function to generate alt text for single image
def generate_alt_text(client, image_base64, device_name=""):
    """Generate alt text using OpenAI Vision API"""
    try:
        start_time = time.time()
        
        default_prompt = "Generate a concise, descriptive alt text for this image that would be useful for screen readers. Focus on the main content and context. Keep it under 125 characters if possible."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": default_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate cost
        cost_info = calculate_cost(response.usage)
        
        # Get generated alt text
        alt_text = response.choices[0].message.content
        
        # Apply device name substitution if provided
        alt_text = substitute_device_name(alt_text, device_name)
        
        # Update session state
        st.session_state.api_calls += 1
        st.session_state.total_cost += cost_info['total_cost']
        st.session_state.total_tokens['input'] += cost_info['input_tokens']
        st.session_state.total_tokens['output'] += cost_info['output_tokens']
        st.session_state.total_execution_time += execution_time
        
        return {
            "alt_text": alt_text,
            "cost": cost_info,
            "execution_time": execution_time
        }
    
    except Exception as e:
        return {
            "alt_text": f"Error: {str(e)}",
            "cost": None,
            "execution_time": 0
        }

# NEW: Optimized function for bulk image processing
def generate_bulk_alt_text(client, images_base64, filenames, device_name=""):
    """
    Generate alt text for multiple images in a single API call.
    This significantly reduces token usage by sharing the prompt context.
    """
    try:
        start_time = time.time()
        
        # Build the content array with one prompt and all images
        content = [
            {
                "type": "text",
                "text": f"""Generate concise, descriptive alt text for each of the {len(images_base64)} images below. 
For each image, provide alt text that would be useful for screen readers, focusing on the main content and context. 
Keep each alt text under 125 characters if possible.

Format your response as a numbered list (1., 2., 3., etc.) with one alt text per line, matching the order of the images."""
            }
        ]
        
        # Add all images to the same message
        for img_base64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=300 * len(images_base64)  # Scale max_tokens with number of images
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate cost
        cost_info = calculate_cost(response.usage)
        
        # Parse the response to extract individual alt texts
        response_text = response.choices[0].message.content
        alt_texts = []
        
        # Split by numbered list items (1., 2., 3., etc.)
        lines = response_text.strip().split('\n')
        for line in lines:
            # Match numbered list format: "1. Alt text here" or "1) Alt text here"
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                alt_text = match.group(1).strip()
                # Apply device name substitution if provided
                alt_text = substitute_device_name(alt_text, device_name)
                alt_texts.append(alt_text)
        
        # Fallback: if parsing failed, split by newlines and clean
        if len(alt_texts) != len(images_base64):
            alt_texts = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            alt_texts = [substitute_device_name(text, device_name) for text in alt_texts]
        
        # Final fallback: if still mismatched, pad or truncate
        if len(alt_texts) < len(images_base64):
            alt_texts.extend([f"Alt text for {filenames[i]}" for i in range(len(alt_texts), len(images_base64))])
        elif len(alt_texts) > len(images_base64):
            alt_texts = alt_texts[:len(images_base64)]
        
        # Update session state (single API call for all images)
        st.session_state.api_calls += 1
        st.session_state.total_cost += cost_info['total_cost']
        st.session_state.total_tokens['input'] += cost_info['input_tokens']
        st.session_state.total_tokens['output'] += cost_info['output_tokens']
        st.session_state.total_execution_time += execution_time
        
        return {
            "alt_texts": alt_texts,
            "cost": cost_info,
            "execution_time": execution_time,
            "success": True
        }
    
    except Exception as e:
        return {
            "alt_texts": [f"Error: {str(e)}"] * len(images_base64),
            "cost": None,
            "execution_time": 0,
            "success": False
        }

# Main app logic
if api_key:
    client = OpenAI(api_key=api_key)
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["Single Image", "Bulk Upload"],
        horizontal=True
    )
    
    # Device name override option
    with st.expander("🔖 Override Device Name (Optional)"):
        device_name = st.text_input(
            "Device name is:",
            placeholder="E.g., VEEV ONE",
            help="Replace any detected device name or generic product term (like 'vape pen') in the generated alt text with this specific name"
        )
    
    st.markdown("---")
    
    # Single Image Mode
    if mode == "Single Image":
        st.subheader("Upload Single Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="single"
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("Generate Alt Text", type="primary", use_container_width=True):
                    with st.spinner("Generating alt text..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        image_base64 = encode_image(uploaded_file)
                        result = generate_alt_text(client, image_base64, device_name)
                        
                        # Store result in session state
                        st.session_state.current_result = result
                        st.rerun()
            
            # Display stored result if it exists
            if st.session_state.current_result:
                result = st.session_state.current_result
                
                st.markdown("---")
                
                # Session statistics in center
                st.subheader("📊 Session Statistics")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("API Calls", st.session_state.api_calls)
                with stat_col2:
                    st.metric("⏱️ Total Time", f"{st.session_state.total_execution_time:.2f}s")
                with stat_col3:
                    st.metric("💰 Total Cost", f"${st.session_state.total_cost:.4f}")
                with stat_col4:
                    if st.session_state.api_calls > 0:
                        avg_cost = st.session_state.total_cost / st.session_state.api_calls
                        st.metric("Avg Cost", f"${avg_cost:.4f}")
                
                st.markdown("---")
                
                st.success("Alt text generated!")
                st.text_area(
                    "Generated Alt Text:",
                    value=result['alt_text'],
                    height=150,
                    key="result_display"
                )
                
                # Character count
                st.caption(f"Character count: {len(result['alt_text'])}")
                
                # Display cost and timing info for this specific call
                if result['cost']:
                    st.markdown("---")
                    st.subheader("This Request")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("⏱️ Execution Time", f"{result['execution_time']:.2f}s")
                    
                    with col_b:
                        st.metric("💰 Cost", f"${result['cost']['total_cost']:.6f}")
                    
                    with col_c:
                        st.metric("🔢 Total Tokens", f"{result['cost']['total_tokens']:,}")
                    
                    with st.expander("📊 Detailed Token Usage"):
                        st.write(f"**Input tokens:** {result['cost']['input_tokens']:,} (${result['cost']['input_cost']:.6f})")
                        st.write(f"**Output tokens:** {result['cost']['output_tokens']:,} (${result['cost']['output_cost']:.6f})")
    
    # Bulk Upload Mode
    else:
        st.subheader("Upload Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="bulk"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} image(s) uploaded")
            
            if st.button("Generate Alt Text for All", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_start_time = time.time()
                
                # Encode all images first
                status_text.text("Encoding images...")
                images_base64 = []
                filenames = []
                
                for file in uploaded_files:
                    file.seek(0)
                    images_base64.append(encode_image(file))
                    filenames.append(file.name)
                
                progress_bar.progress(0.3)
                
                # Generate alt text for all images in a single API call
                status_text.text(f"Generating alt text for {len(uploaded_files)} images in a single optimized request...")
                
                bulk_result = generate_bulk_alt_text(client, images_base64, filenames, device_name)
                
                progress_bar.progress(0.9)
                
                # Build results list
                results = []
                if bulk_result['success']:
                    for idx, (filename, alt_text) in enumerate(zip(filenames, bulk_result['alt_texts'])):
                        # Calculate per-image cost (approximation by dividing total cost)
                        per_image_cost = bulk_result['cost']['total_cost'] / len(filenames) if bulk_result['cost'] else 0.0
                        per_image_tokens = bulk_result['cost']['total_tokens'] // len(filenames) if bulk_result['cost'] else 0
                        
                        results.append({
                            "Filename": filename,
                            "Alt Text": alt_text,
                            "Character Count": len(alt_text),
                            "Est. Cost ($)": f"{per_image_cost:.6f}",
                            "Est. Tokens": per_image_tokens
                        })
                else:
                    # Fallback to individual processing if batch fails
                    status_text.text("Batch processing failed, falling back to individual processing...")
                    for idx, (file, filename) in enumerate(zip(uploaded_files, filenames)):
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {filename}")
                        
                        file.seek(0)
                        image_base64 = encode_image(file)
                        result = generate_alt_text(client, image_base64, device_name)
                        
                        cost = result['cost']['total_cost'] if result['cost'] else 0.0
                        
                        results.append({
                            "Filename": filename,
                            "Alt Text": result['alt_text'],
                            "Character Count": len(result['alt_text']),
                            "Est. Cost ($)": f"{cost:.6f}",
                            "Est. Tokens": result['cost']['total_tokens'] if result['cost'] else 0
                        })
                        
                        progress_bar.progress(0.3 + (0.6 * (idx + 1) / len(uploaded_files)))
                
                batch_end_time = time.time()
                total_batch_time = batch_end_time - batch_start_time
                batch_cost = bulk_result['cost']['total_cost'] if bulk_result['cost'] else 0.0
                
                progress_bar.progress(1.0)
                status_text.text("✅ All images processed!")
                
                # Store results in session state
                st.session_state.bulk_results = {
                    'results': results,
                    'total_batch_time': total_batch_time,
                    'batch_cost': batch_cost,
                    'uploaded_files': uploaded_files,
                    'is_optimized': bulk_result['success']
                }
                st.rerun()
        
        # Display stored bulk results if they exist
        if st.session_state.bulk_results:
            bulk_data = st.session_state.bulk_results
            results = bulk_data['results']
            total_batch_time = bulk_data['total_batch_time']
            batch_cost = bulk_data['batch_cost']
            is_optimized = bulk_data.get('is_optimized', False)
            
            # Show optimization indicator
            if is_optimized:
                st.success("✨ Optimized batch processing used - reduced token usage!")
            
            # Session statistics in center
            st.markdown("---")
            st.subheader("📊 Session Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("API Calls", st.session_state.api_calls)
            with stat_col2:
                st.metric("⏱️ Total Time", f"{st.session_state.total_execution_time:.2f}s")
            with stat_col3:
                st.metric("💰 Total Cost", f"${st.session_state.total_cost:.4f}")
            with stat_col4:
                if st.session_state.api_calls > 0:
                    avg_cost = st.session_state.total_cost / st.session_state.api_calls
                    st.metric("Avg Cost", f"${avg_cost:.4f}")
            
            # Batch summary metrics
            st.markdown("---")
            st.subheader("📊 This Batch")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Images Processed", len(results))
            with col2:
                st.metric("Total Time", f"{total_batch_time:.2f}s")
            with col3:
                st.metric("Avg Time/Image", f"{total_batch_time/len(results):.2f}s")
            with col4:
                st.metric("Batch Cost", f"${batch_cost:.4f}")
            
            # Display results
            st.markdown("---")
            st.subheader("Results")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"alt_text_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Show preview of images with alt text
            st.markdown("---")
            st.subheader("Preview")
            
            # Get the uploaded files from session state
            if 'uploaded_files' in bulk_data:
                uploaded_files_list = bulk_data['uploaded_files']
                cols = st.columns(3)
                for idx, (file, result) in enumerate(zip(uploaded_files_list, results)):
                    with cols[idx % 3]:
                        file.seek(0)
                        st.image(file, caption=file.name, use_container_width=True)
                        st.caption(f"**Alt:** {result['Alt Text']}")
                        st.caption(f"💰 {result['Est. Cost ($)']}")
                        st.markdown("---")

else:
    st.info("👈 Please enter your OpenAI API key in the sidebar to get started")
    
    st.markdown("### How to use:")
    st.markdown("""
    1. Get your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
    2. Enter it in the sidebar
    3. Upload single or multiple images
    4. (Optional) Specify a device name to override generic terms
    5. Generate alt text automatically
    6. Track costs and execution times
    7. Download results as CSV (bulk mode)
    
    **✨ New:** Bulk uploads now use optimized batch processing to reduce token usage!
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Alt-Text Wizard 🧙‍♂️✨ – Powered by OpenAI GPT-4o Vision</div>",
    unsafe_allow_html=True
)