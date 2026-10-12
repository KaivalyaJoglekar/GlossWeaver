import { useEffect, useState } from 'react';
import { fetchDatasets } from '../api';

export default function DataPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    fetchDatasets().then(value => { if (active) setData(value); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  return <div className="dataset-panel">
    <div className="page-header"><h1 className="page-title">Your datasets</h1><p className="page-subtitle">Stored locally in the project. Raw downloads are in datasets/raw; cleaned splits are in datasets/processed.</p></div>
    <p className="data-location">GlossWeaver / datasets /</p>
    {error && <p role="alert" className="error-banner">{error}</p>}
    {!data && !error && <p role="status" className="form-hint">Reading dataset manifests…</p>}
    {data && <div className="dataset-table-scroll"><table className="data-table"><thead><tr><th>Dataset</th><th>Prepared splits</th><th>Purpose</th></tr></thead><tbody>{data.datasets.map(row => <tr key={row.folder}>
      <td><strong>{row.name}</strong><p className="form-hint">{row.kind}</p><code>{row.folder}</code></td>
      <td>{row.available ? Object.entries(row.counts).map(([split, count]) => <p key={split}>{split.replaceAll('_', ' ')} <span className="quiet-label">{count.toLocaleString()}</span></p>) : 'Not prepared'}</td>
      <td>{row.usage}<p className="form-hint">{row.license}</p></td>
    </tr>)}</tbody></table></div>}
    <p className="form-hint">Open datasets/README.md in your project for provenance and preprocessing instructions. ASLG is rule-generated; Flores contains human annotations. Synthetic scores do not establish real ASL performance.</p>
  </div>;
}
