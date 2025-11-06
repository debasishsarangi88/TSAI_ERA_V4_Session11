# Quick Start Guide - Hugging Face Deployment

Deploy your Odia BPE Tokenizer to Hugging Face Spaces in 5 minutes.

## Prerequisites

- A Hugging Face account (free registration at huggingface.co)
- All files from this `huggingface_files` folder

## Deployment Steps

### Step 1: Create a New Space

1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure your space:
   - **Space name**: `odia-bpe-tokenizer` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free tier)
   - **Visibility**: Public
3. Click "Create Space"

### Step 2: Upload Required Files

Navigate to "Files and versions" → "Add file" → "Upload files"

Upload these 6 files:

```
app.py                          (Main application)
requirements.txt                (Dependencies)
README.md                       (Space description)
.gitattributes                  (Git LFS configuration)
odia_bpe_tokenizer_4500.pkl    (Trained model - essential)
training_results_4500.json     (Training statistics)
```

**Note**: Ensure the `.pkl` file is uploaded correctly as it contains the trained model.

### Step 3: Build and Deploy

- The Space will automatically build (approximately 2-3 minutes)
- Monitor build progress in the "Building" tab
- Once complete, your app will be live at:
  `https://huggingface.co/spaces/YOUR_USERNAME/odia-bpe-tokenizer`

## Testing Your Deployment

Test with these Odia text samples:

1. Greeting: `ନମସ୍କାର ! ଆପଣ କେମିତି ଅଛନ୍ତି ?`
2. About Odisha: `ଓଡ଼ିଶା ଭାରତର ପୂର୍ବ ଉପକୂଳରେ ଅବସ୍ଥିତ ।`
3. Education: `ଶିକ୍ଷା ହିଁ ଜୀବନର ପ୍ରକୃତ ସମ୍ପତ୍ତି ।`

## Expected Features

Your deployed Space will display:

- Token Count: 3,500 (requirement: < 5,000)
- Compression Ratio: 20.16 (requirement: ≥ 3.2)
- Interactive tokenizer with real-time encoding/decoding
- Pre-loaded example texts
- Performance statistics

## Optional Customization

You can personalize your Space by editing the configuration:

### Modify Space Appearance

Edit the YAML front matter in `README.md`:

```yaml
---
title: Your Custom Title
emoji: 🎯
colorFrom: blue
colorTo: purple
---
```

### Update GitHub Link

In `app.py`, locate and update:

```python
**GitHub**: [View Source Code](https://github.com/yourusername/odia-bpe-tokenizer)
```

## Troubleshooting

### Space Failed to Build
**Solution**: Verify all 6 files are uploaded, particularly the `.pkl` file

### Cannot Load Tokenizer
**Solution**: Ensure `odia_bpe_tokenizer_4500.pkl` is in the root directory

### Module Not Found Error
**Solution**: Confirm `requirements.txt` contains `gradio==4.10.0`

## Support Resources

- Hugging Face Documentation: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- Gradio Documentation: [gradio.app/docs](https://gradio.app/docs/)
- Hugging Face Discord Community: [hf.co/join/discord](https://hf.co/join/discord)

## Requirements Verification

Your deployed Space will clearly show:

| Requirement | Target | Achieved | Status |
|-------------|--------|----------|--------|
| Token Count | < 5,000 | 3,500 | Pass |
| Compression Ratio | ≥ 3.2 | 20.16 | Pass |

Both requirements are significantly exceeded.

## Sharing Your Deployment

Once deployed, you can share your Space with:

- Academic community and instructors
- Odia language researchers and enthusiasts
- Natural Language Processing communities
- Include in your professional portfolio

Your tokenizer is now accessible at: `https://huggingface.co/spaces/YOUR_USERNAME/odia-bpe-tokenizer`

