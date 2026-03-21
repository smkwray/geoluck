import fs from "node:fs";
import path from "node:path";

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function bundleSpecMap(summaryPayload) {
  const map = new Map();
  for (const target of summaryPayload.targets ?? []) {
    for (const bundle of target.bundles ?? []) {
      map.set(`${target.target}::${bundle.feature_tier}`, bundle.spec_name);
    }
  }
  return map;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const publicDataDir = path.resolve("public", "data");
const distDataDir = path.resolve("dist", "data");

const publicManifestPath = path.join(publicDataDir, "data_manifest.json");
const distManifestPath = path.join(distDataDir, "data_manifest.json");
const publicMetadataPath = path.join(publicDataDir, "metadata.json");
const distMetadataPath = path.join(distDataDir, "metadata.json");
const publicSummaryPath = path.join(publicDataDir, "bundle_summary.json");
const distSummaryPath = path.join(distDataDir, "bundle_summary.json");

for (const requiredPath of [
  publicManifestPath,
  distManifestPath,
  publicMetadataPath,
  distMetadataPath,
  publicSummaryPath,
  distSummaryPath,
]) {
  assert(fs.existsSync(requiredPath), `Missing required file: ${requiredPath}`);
}

const publicManifest = readJson(publicManifestPath);
const distManifest = readJson(distManifestPath);
const publicMetadata = readJson(publicMetadataPath);
const distMetadata = readJson(distMetadataPath);
const publicSummary = readJson(publicSummaryPath);
const distSummary = readJson(distSummaryPath);

assert(
  publicManifest.export_id === distManifest.export_id,
  "public/data and dist/data disagree on export_id",
);
assert(
  publicManifest.payload_version === distManifest.payload_version,
  "public/data and dist/data disagree on payload_version",
);
assert(
  publicMetadata.data_export_id === publicManifest.export_id,
  "public metadata export id does not match public data manifest",
);
assert(
  distMetadata.data_export_id === distManifest.export_id,
  "dist metadata export id does not match dist data manifest",
);

const publicFiles = new Set((publicManifest.files ?? []).map((entry) => entry.path));
const distFiles = new Set((distManifest.files ?? []).map((entry) => entry.path));
for (const requiredName of [
  "bundle_summary.json",
  "bundle_feature_effects.json",
  "bundle_permutation_importance.json",
  "bundle_country_contributions_index.json",
]) {
  assert(publicFiles.has(requiredName), `public manifest missing ${requiredName}`);
  assert(distFiles.has(requiredName), `dist manifest missing ${requiredName}`);
}

const publicSpecs = bundleSpecMap(publicSummary);
const distSpecs = bundleSpecMap(distSummary);
assert(
  publicSpecs.size === distSpecs.size,
  "public and dist bundle summary payloads have different bundle counts",
);

for (const [key, specName] of publicSpecs) {
  assert(distSpecs.has(key), `dist bundle summary missing ${key}`);
  assert(
    distSpecs.get(key) === specName,
    `bundle summary display spec mismatch for ${key}: ${specName} vs ${distSpecs.get(key)}`,
  );
}

console.log(
  `Verified dist data against public data (${publicManifest.export_id.slice(0, 12)})`,
);
