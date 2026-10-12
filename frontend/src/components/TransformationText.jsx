export default function TransformationText({ result }) {
  return <div className="resolved-sentence"><p className="resolution-source" aria-hidden="true">{result.gloss}</p><p className="result-text">{result.reconstruction}</p></div>;
}
