# 🎨 Creative Flow Tracker - Quick Start

Track your creative states and discover your patterns using Grounded Coherence Theory (GCT).

## 🚀 Start in 30 Seconds

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/GreatPyreneseDad/GCT.git
cd GCT/gct-creative-flow

# Launch the app
./start.sh
```

That's it! The app will open in your browser.

## 📱 How to Use

### 1. Quick Entry (Easy Mode)
- Click a **mood button** that matches how you feel creatively
- Add what project you're working on (optional)
- Hit **Save Entry**
- That's it! 

### 2. See Your Patterns
- Go to **My Patterns** tab
- View your creative states over time
- Discover when you're most creative
- See which projects flow best

### 3. Track Progress
- Check your **coherence trend**
- See your **flow achievement**
- Export your data anytime

## 🎯 What the Numbers Mean

- **Coherence**: Overall creative harmony (0-1)
  - 0.75+ = Flow state 🌊
  - 0.5-0.75 = Good creative energy
  - Below 0.5 = Need a break or change
  
- **Creative States**:
  - 🌊 **Flow** - In the zone!
  - 💡 **Illumination** - Having breakthroughs
  - 🔍 **Exploration** - Finding new ideas
  - 🌙 **Incubation** - Processing in background
  - 🚧 **Blocked** - Time for a walk!

## 💾 Your Data

- All data is stored locally in `creative_flow_data.json`
- Export to CSV anytime
- No cloud, no tracking, just your creative journey

## 🎨 Creative Mood Buttons

- 🔥 **On Fire** - Everything is clicking
- 💡 **Inspired** - Full of ideas
- 🌊 **Flowing** - Smooth and effortless
- 🤔 **Exploring** - Trying new things
- 😴 **Incubating** - Let it simmer
- 😤 **Blocked** - Stuck (we've all been there)
- 🎯 **Focused** - Laser concentration
- 🎨 **Playful** - Fun and experimental

## 📊 Tips for Best Results

1. **Track daily** - Even just one entry helps
2. **Be honest** - There's no "wrong" state
3. **Note breakthroughs** - Use the notes field
4. **Check patterns** - Learn what works for you
5. **Experiment** - Try different times/environments

## 🔧 Advanced Features

### Custom Input
Use the sliders for precise control:
- **Clarity** - How clear is your thinking?
- **Depth** - How deep are your insights?
- **Energy** - How emotionally engaged?
- **Connection** - How connected do you feel?

### Goals & Tracking
- Set coherence goals
- Track your streak
- Get milestone achievements

### Export Options
- CSV for spreadsheets
- Text reports for documentation
- JSON for developers

## 🐛 Troubleshooting

**App won't start?**
```bash
# Manual install
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit plotly numpy pandas scipy networkx
streamlit run app_simple.py
```

**No module found?**
Make sure you're in the `gct-creative-flow` directory.

**Data not saving?**
Check that `creative_flow_data.json` has write permissions.

## 🎯 Quick Experiments to Try

1. **Morning vs Evening**: Track at different times
2. **Music Test**: Try with/without music
3. **Location Change**: Coffee shop vs home
4. **Project Switching**: Which projects flow better?
5. **Break Patterns**: How do breaks affect coherence?

---

Made with ❤️ using Grounded Coherence Theory

*"Creativity is allowing yourself to make mistakes. Art is knowing which ones to keep."*