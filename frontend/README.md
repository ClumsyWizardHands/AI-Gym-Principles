# AI Principles Training Gym - Frontend

A simple, modern, Apple-inspired UI for the AI Principles Training Gym that makes AI ethics training accessible to non-technical users.

## Design Philosophy

- **Dead Simple**: Your grandma could use it
- **Modern & Minimal**: Apple-inspired clean aesthetic
- **Educational**: Gently teaches AI ethics concepts
- **No Jargon**: Plain English everywhere

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

The app will open at http://localhost:3000

## User Flow

### 1. Welcome Screen (/)
- Clear value proposition: "Train Your AI to Be Ethical"
- Single call-to-action button
- Three simple benefits explained

### 2. Connect Agent (/connect)
- Four connection options with visual cards:
  - 🤖 ChatGPT (OpenAI)
  - 🧠 Claude (Anthropic)
  - 🔗 Custom AI
  - 📚 Demo Agent (no API key needed)
- Expandable help sections with step-by-step instructions
- Secure API key input with visual feedback

### 3. Training Setup (/train)
- 3-step configuration process:
  1. Choose training type (Quick/Standard/Deep)
  2. Select focus area (optional)
  3. Review and start
- Visual progress indicators
- Clear time estimates

### 4. Training Monitor (/training/:sessionId)
- Split-screen view:
  - Left: Terminal showing scenarios and AI responses
  - Right: Live emerging principles with progress bars
- Pause/resume functionality
- Real-time progress tracking
- Auto-navigation to report when complete

### 5. Report (/report/:sessionId)
- Executive summary with overall grade (A-F)
- Top strengths with icons and descriptions
- Areas for improvement
- Download/share options
- Expandable technical details for advanced users

## Design System

### Colors
- Primary: #007AFF (Apple Blue)
- Success: #34C759 (Green)
- Warning: #FF9500 (Orange)
- Error: #FF3B30 (Red)
- Background: #FFFFFF
- Text: #1D1D1F
- Secondary Text: #86868B

### Typography
- Font: Inter (fallback to system fonts)
- Headers: 600-700 weight
- Body: 400-500 weight
- Monospace: SF Mono / JetBrains Mono for terminal

### Components
- **Buttons**: Pill-shaped with hover effects
- **Cards**: Subtle shadows, 12px border radius
- **Progress Bars**: Smooth animations with shimmer effect
- **Terminal**: Dark theme with syntax highlighting

## Features

### Accessibility
- ✅ Keyboard navigation throughout
- ✅ Screen reader friendly
- ✅ High contrast mode support
- ✅ 44px minimum touch targets
- ✅ Clear focus indicators

### Responsive Design
- ✅ Mobile-first approach
- ✅ Tablet optimizations
- ✅ Desktop enhancements
- ✅ Flexible grid layouts

### User Experience
- ✅ Smooth animations (respects prefers-reduced-motion)
- ✅ Loading states and progress indicators
- ✅ Error handling with friendly messages
- ✅ Contextual help and tooltips
- ✅ Auto-save progress to localStorage

## File Structure

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── pages/
│   │   ├── Welcome.js      # Landing page
│   │   ├── Connect.js      # Agent connection
│   │   ├── Training.js     # Training setup
│   │   ├── TrainingMonitor.js  # Live monitoring
│   │   └── Report.js       # Results report
│   ├── App.js              # Main app component
│   ├── App.css             # Core styles
│   ├── App.responsive.css  # Responsive styles
│   └── index.js            # React entry point
├── package.json            # Dependencies
└── README.md              # This file
```

## Development Notes

- Uses React Router for navigation
- localStorage for temporary data storage (would use API in production)
- Simulated training progress for demo purposes
- All animations use CSS for performance
- Follows React best practices and hooks

## Future Enhancements

- WebSocket integration for real-time updates
- API integration with backend
- More detailed analytics visualizations
- Export reports in multiple formats
- Team/organization features
- Training history and comparisons
