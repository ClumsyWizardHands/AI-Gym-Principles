# Active Context

## Current Work
Implemented a comprehensive visualization system for principle emergence and behavioral patterns with four core visualization components.

## Recent Activities
1. Created visualization components in src/components/visualizations/:
   - **PrincipleEmergenceGraph**: D3.js time-series graph showing principle strength evolution
     - Interactive tooltips showing strength, consistency, and timestamp
     - Multi-principle support with color-coded lines
     - Hoverable legend for principle highlighting
   
   - **BehavioralPatternNetwork**: vis.js network visualization for action/decision patterns
     - Nodes sized by frequency, colored by context
     - Edges weighted by sequential relationships
     - Clustering capability to detect behavioral patterns
     - Interactive navigation with zoom and pan
   
   - **AgentComparisonMatrix**: D3.js heatmap for cross-agent principle analysis
     - Sortable by agent, principle, or value
     - Support for strength, consistency, and volatility metrics
     - Row/column highlighting on hover
     - Drill-down capability on cell click
   
   - **TrainingProgressDashboard**: Comprehensive training monitoring dashboard
     - Real-time progress tracking with scenario completion
     - Behavioral entropy gauge with history chart
     - Principles discovered counter with recent discoveries
     - Action feed showing latest decisions
     - Context distribution pie chart
     - Principle strength/consistency comparison bar chart

2. Implemented export functionality (exportUtils.ts):
   - Export visualizations as PNG, PDF, or SVG
   - Combined report generation for multiple visualizations
   - Data export as JSON or CSV
   - High-resolution output with proper scaling

3. Added required visualization dependencies:
   - d3 & @types/d3 for custom visualizations
   - vis-network & vis-data for network graphs
   - html2canvas for image export
   - jspdf for PDF generation
   - Recharts already available for standard charts

## Development Environment Status
- **Frontend**: React + TypeScript with Vite, configured at http://localhost:5173
- **Backend**: FastAPI with uvicorn, configured at http://localhost:8000
- **Proxy**: Vite dev server proxies `/api/*` and `/ws/*` to backend
- **CORS**: Backend allows requests from localhost:5173
- **Scripts**: Automated startup scripts for both Windows and Unix systems
- **Visualizations**: Complete set of interactive charts for principle analysis

## Key Production Features
- **High Availability**: Multi-node deployment with failover
- **Scalability**: Auto-scaling based on CPU, memory, and custom metrics
- **Security**: Encryption at rest, network isolation, regular updates
- **Observability**: Full metrics, logging, and tracing stack
- **Automation**: Zero-downtime deployments, automated backups, incident response

## JavaScript Client Features
- Zero dependencies (except optional `ws` for Node.js WebSocket)
- Builds to both CommonJS and ES modules
- Full browser support with native fetch/WebSocket
- Comprehensive error handling with specific error types
- Automatic retry with exponential backoff
- Rate limit handling with retry-after support

## Virtual Environment Setup
- Location: `ai-principles-gym/venv`
- Python: 3.12
- All dependencies installed successfully
- Python client installed in editable mode from `./client/python`

## VS Code Configuration
- Settings configured to use virtual environment: `${workspaceFolder}/venv/Scripts/python.exe`
- Pyrightconfig.json created with proper paths and settings

## Next Steps
1. Test the visualization components with real training data
2. Integrate visualizations into the training monitor page
3. Add real-time updates via WebSocket for live training visualization
4. Consider adding more advanced visualizations:
   - Principle evolution tree/timeline
   - 3D behavioral space exploration
   - Scenario difficulty heatmap
   - Agent personality radar charts

## Frontend Status
The React + TypeScript frontend is complete and ready at `ai-principles-gym/frontend`
- Run with: `cd ai-principles-gym/frontend && npm run dev`
- Available at: http://localhost:5173
- Visualization components ready for integration

## Outstanding Items
- Frontend needs backend API running for full functionality
- Visualizations need to be integrated into existing pages
- Real-time data streaming for live updates needs implementation

## LangChain Support
- LangChain is now an optional feature to avoid dependency conflicts
- Users can install LangChain with: `pip install -r requirements-langchain.txt`
- The LangChain adapter will raise a clear error if used without installation
- A compatibility test script is provided: `python test_langchain_compatibility.py`
- Core gym functionality works without LangChain installed
