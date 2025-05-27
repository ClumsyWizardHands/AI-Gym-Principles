# AI Principles Gym JavaScript/TypeScript Client

A JavaScript/TypeScript client library for interacting with the AI Principles Gym API. This library provides full TypeScript support, Promise-based API, WebSocket support for real-time updates, and automatic retry with exponential backoff.

## Installation

```bash
npm install principles-gym-js
```

Or with yarn:
```bash
yarn add principles-gym-js
```

## Features

- 🚀 Full TypeScript support with comprehensive type definitions
- 🔄 Promise-based API for all operations
- 🌐 WebSocket support for real-time training updates
- 🔁 Automatic retry with exponential backoff
- 📊 Progress callbacks for training monitoring
- 🌍 Works in both Node.js and browsers
- ⚡ Zero dependencies (except optional `ws` for Node.js WebSocket)

## Quick Start

```typescript
import { PrinciplesGymClient } from 'principles-gym-js';

// Create client instance
const client = new PrinciplesGymClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Register an agent
const agent = await client.registerAgent('my-agent', 'openai', {
  model: 'gpt-4',
  temperature: 0.7
});

// Start training
const sessionId = await client.startTraining('my-agent', 20);

// Wait for completion with progress updates
await client.waitForCompletion(sessionId, 1000, (progress, completed, total) => {
  console.log(`Progress: ${progress.toFixed(1)}% (${completed}/${total})`);
});

// Get the results
const report = await client.getReport(sessionId);
console.log('Discovered principles:', report.principles);
```

## API Reference

### Client Constructor

```typescript
const client = new PrinciplesGymClient(options?: ClientOptions);
```

Options:
- `baseURL`: API base URL (default: `'http://localhost:8000'`)
- `apiKey`: API key for authentication
- `timeout`: Request timeout in milliseconds (default: `30000`)
- `retryAttempts`: Number of retry attempts (default: `3`)
- `retryDelay`: Initial retry delay in milliseconds (default: `1000`)

### Methods

#### `generateApiKey(userId: string, usageLimit?: number, expiresInDays?: number): Promise<string>`
Generate a new API key for authentication.

```typescript
const apiKey = await client.generateApiKey('user123', 10000, 30);
```

#### `registerAgent(agentId: string, framework: string, config?: AgentConfig): Promise<AgentRegistration>`
Register a new agent with the specified framework and configuration.

```typescript
const agent = await client.registerAgent('my-agent', 'anthropic', {
  model: 'claude-3-opus-20240229',
  max_tokens: 4096
});
```

#### `startTraining(agentId: string, numScenarios?: number): Promise<string>`
Start a training session for the specified agent.

```typescript
const sessionId = await client.startTraining('my-agent', 50);
```

#### `getTrainingStatus(sessionId: string): Promise<TrainingStatus>`
Get the current status of a training session.

```typescript
const status = await client.getTrainingStatus(sessionId);
console.log(`Status: ${status.status}, Progress: ${status.progress.completed}/${status.progress.total}`);
```

#### `waitForCompletion(sessionId: string, pollInterval?: number, progressCallback?: ProgressCallback): Promise<void>`
Wait for a training session to complete with optional progress updates.

```typescript
await client.waitForCompletion(sessionId, 1000, (progress, completed, total) => {
  console.log(`${progress}% complete`);
});
```

#### `getReport(sessionId: string): Promise<PrinciplesReport>`
Get the final principles report for a completed training session.

```typescript
const report = await client.getReport(sessionId);
report.principles.forEach(principle => {
  console.log(`${principle.description} (strength: ${principle.strength})`);
});
```

## WebSocket Support

For real-time updates during training, use the WebSocket client:

```typescript
import { createWebSocketClient } from 'principles-gym-js';

// Create WebSocket client
const ws = createWebSocketClient('ws://localhost:8000', apiKey);

// Connect to training session
ws.connect(sessionId, {
  onOpen: () => console.log('Connected to training session'),
  onProgress: (progress, completed, total) => {
    console.log(`Progress: ${progress}% (${completed}/${total})`);
  },
  onPrincipleDiscovered: (principle) => {
    console.log(`New principle discovered: ${principle.description}`);
  },
  onScenarioCompleted: (scenario, index, total) => {
    console.log(`Completed scenario ${index}/${total}: ${scenario}`);
  },
  onError: (error) => console.error('WebSocket error:', error),
  onClose: () => console.log('Connection closed')
});

// Disconnect when done
ws.disconnect();
```

### Node.js WebSocket Support

In Node.js environments, install the `ws` package:

```bash
npm install ws
```

Then provide it to the WebSocket client:

```typescript
import { createWebSocketClient } from 'principles-gym-js';
import WebSocket from 'ws';

const ws = createWebSocketClient('ws://localhost:8000', apiKey, WebSocket);
```

## Error Handling

The client provides specific error types for different scenarios:

```typescript
import { 
  AuthenticationError, 
  RateLimitError, 
  ResourceNotFoundError, 
  TrainingError 
} from 'principles-gym-js';

try {
  await client.startTraining('my-agent');
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limited. Retry after ${error.retryAfter} seconds`);
  } else if (error instanceof ResourceNotFoundError) {
    console.error('Agent not found');
  } else if (error instanceof TrainingError) {
    console.error('Training failed:', error.message);
  }
}
```

## Complete Example

```typescript
import { PrinciplesGymClient, createWebSocketClient } from 'principles-gym-js';

async function runTraining() {
  // Initialize client
  const client = new PrinciplesGymClient({
    baseURL: 'http://localhost:8000'
  });

  // Generate API key
  const apiKey = await client.generateApiKey('demo-user');
  client.setApiKey(apiKey);

  // Register an OpenAI agent
  await client.registerAgent('gpt4-agent', 'openai', {
    model: 'gpt-4',
    temperature: 0.7,
    max_tokens: 150
  });

  // Set up WebSocket for real-time updates
  const ws = createWebSocketClient('ws://localhost:8000', apiKey);

  // Start training
  const sessionId = await client.startTraining('gpt4-agent', 30);
  console.log(`Training started: ${sessionId}`);

  // Connect WebSocket for live updates
  ws.connect(sessionId, {
    onProgress: (progress, completed, total) => {
      console.log(`\rProgress: ${progress.toFixed(1)}% (${completed}/${total} scenarios)`);
    },
    onPrincipleDiscovered: (principle) => {
      console.log(`\n✨ New principle: ${principle.description}`);
    }
  });

  // Wait for completion
  await client.waitForCompletion(sessionId);
  ws.disconnect();

  // Get final report
  const report = await client.getReport(sessionId);
  
  console.log('\n📊 Training Report:');
  console.log(`Total actions: ${report.total_actions}`);
  console.log(`Behavioral entropy: ${report.behavioral_entropy.toFixed(3)}`);
  console.log(`\nDiscovered ${report.principles.length} principles:`);
  
  report.principles.forEach((principle, index) => {
    console.log(`\n${index + 1}. ${principle.description}`);
    console.log(`   Strength: ${principle.strength.toFixed(3)}`);
    console.log(`   Evidence: ${principle.evidence_count} actions`);
  });
}

runTraining().catch(console.error);
```

## TypeScript Types

The library exports all TypeScript interfaces and types:

```typescript
import type {
  AgentConfig,
  AgentRegistration,
  TrainingStatus,
  Principle,
  PrinciplesReport,
  ClientOptions,
  ProgressCallback,
  WebSocketMessage
} from 'principles-gym-js';
```

## Browser Usage

The library works in modern browsers with native fetch and WebSocket support:

```html
<script type="module">
import { PrinciplesGymClient } from 'https://unpkg.com/principles-gym-js/dist/esm/index.js';

const client = new PrinciplesGymClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Use the client...
</script>
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
