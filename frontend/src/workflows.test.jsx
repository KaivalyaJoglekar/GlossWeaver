import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import InferencePanel from './components/InferencePanel';
import ComparisonPanel from './components/ComparisonPanel';
import MetricsPanel from './components/MetricsPanel';
import * as api from './api';
vi.mock('./api');
afterEach(() => { cleanup(); vi.resetAllMocks(); });
const models = [{ checkpoint: 'balanced-flan', name: 'Balanced model', available: true, default: true, architecture: 'FLAN-T5-small', debug: { model_type: 'flan-t5' } }, { checkpoint: 'missing', name: 'Missing model', available: false, debug: { model_type: 'flan-t5' } }];
const result = { gloss: 'ME GO', reconstruction: 'I go.', elapsed_ms: 10, checkpoint: 'balanced-flan', model_identifier: 'local/run', debug: { model_name: 'Balanced model' }, grammar_aware: false, notices: [], alternatives: [], changes: [], semantic: { issues: [], checks: [], advisory: 'Advisory only' } };
function setup() {
  api.fetchModels.mockResolvedValue({ models }); api.fetchDatasets.mockResolvedValue({ datasets: [] });
  api.fetchHealth.mockResolvedValue({ status: 'ok' }); api.fetchMetrics.mockResolvedValue({ rows: [], evaluations: [], provenance: 'Saved' });
}
describe('laboratory workflows', () => {
  it('renders Reconstruction Studio directly in App', async () => {
    setup(); render(<App />);
    expect(screen.getByLabelText('Text to reconstruct')).toBeTruthy();
    await waitFor(() => expect(screen.getByRole('option', { name: 'Balanced model' })).toBeTruthy());
  });
  it('validates blank input, submits with settings, and renders diagnostics', async () => {
    setup(); api.runInfer.mockResolvedValue(result); const user = userEvent.setup(); render(<InferencePanel />);
    const button = screen.getByRole('button', { name: 'Reconstruct sentence' });
    expect(button.disabled).toBe(true);
    await user.type(screen.getByLabelText('Text to reconstruct'), 'ME GO');
    await waitFor(() => expect(button.disabled).toBe(false));
    await user.click(button);
    await screen.findAllByText('I go.');
    expect(api.runInfer).toHaveBeenCalledWith(expect.objectContaining({ gloss: 'ME GO', checkpoint: 'balanced-flan', alternatives: 1 }));
    expect(screen.getByText('No issue detected')).toBeTruthy();
    expect(screen.queryByRole('option', { name: /Missing model/ })).toBeNull();
  });
  it('shows a loading state and an actionable API error', async () => {
    setup(); let reject;
    api.runInfer.mockImplementation(() => new Promise((resolve, rejectRun) => { reject = rejectRun; }));
    const user = userEvent.setup(); render(<InferencePanel />);
    await user.click(screen.getByRole('button', { name: 'Past tense' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reconstruct sentence' }).disabled).toBe(false));
    await user.click(screen.getByRole('button', { name: 'Reconstruct sentence' }));
    expect(screen.getByRole('button', { name: 'Reconstructing…' }).disabled).toBe(true);
    reject(new Error('Checkpoint could not load'));
    expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'Checkpoint could not load');
  });
  it('compares models using the shared input and renders partial failures', async () => {
    api.runCompare.mockResolvedValue({ results: [{ model: 'balanced-flan', status: 'ok', result }, { model: 'missing', status: 'error', error: 'Unavailable' }] });
    const user = userEvent.setup(); render(<ComparisonPanel models={models} />);
    await user.type(screen.getByLabelText('Shared input'), 'ME GO');
    await user.click(screen.getByRole('checkbox', { name: /Balanced model/ }));
    await user.click(screen.getByRole('button', { name: 'Compare models →' }));
    expect(await screen.findByText('I go.')).toBeTruthy();
    expect(screen.getByText('Unavailable')).toBeTruthy();
    expect(api.runCompare).toHaveBeenCalledWith(expect.objectContaining({ gloss: 'ME GO', num_beams: 4, models: ['balanced-flan'] }));
  });
  it('keeps keyboard-triggered reconstruction results instant', async () => {
    setup(); api.runInfer.mockResolvedValue(result);
    const user = userEvent.setup(); const { container } = render(<InferencePanel />);
    await user.type(screen.getByLabelText('Text to reconstruct'), 'ME GO');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reconstruct sentence' }).disabled).toBe(false));
    await user.keyboard('{Control>}{Enter}{/Control}');
    await screen.findAllByText('I go.');
    expect(container.querySelector('.result-view').getAttribute('data-instant')).toBe('true');
  });
  it('loads the initial metric selection without runtime errors', async () => {
    setup(); api.fetchMetrics.mockResolvedValue({ rows: [{ model: 'balanced-flan', display_name: 'Balanced model', dataset: 'fixture', split: 'dev', examples: 1, bleu: 42 }], evaluations: [{ dataset: 'fixture', split: 'dev' }], provenance: 'Saved' });
    render(<MetricsPanel />);
    expect(await screen.findByText('42.00')).toBeTruthy();
  });
});
