import {
  optionalNumber,
  requireArray,
  requireNumber,
  requireRecord,
  requireString,
} from './transport.ts';

export type DistinguishabilityRecord = {
  kind: string;
  metric_id?: string;
  resolution?: number;
  certificate_id?: string;
};

export type StateSpaceAmbientRecord = {
  field_domain_kind: string;
  field_domain: Record<string, string | number>;
  field_codomain_id: string;
  distinguishability: DistinguishabilityRecord;
};

export type AxisDomainRecord =
  | { kind: 'integer-range'; lower: number; upper: number }
  | { kind: 'real-grid'; lower: number; upper: number; count: number }
  | { kind: 'enumerated-cells'; cells: string[] }
  | { kind: 'binary-vector'; dimension: number };

export type StateSpaceAxisRecord = {
  id: string;
  domain: AxisDomainRecord;
};

export type AxisRegionRecord = {
  axis: StateSpaceAxisRecord;
  coordinate_region: (number | string)[];
  count: number;
  log2_count: number;
};

export type ProductRegionRecord = {
  axis_regions: AxisRegionRecord[];
  measure_rule: string;
  volume: number;
  log2_volume: number;
  stratum_id?: string;
  stratum_target?: Record<string, unknown>;
};

export type StateSpaceRegionRecord = {
  id: string;
  ambient: StateSpaceAmbientRecord;
  components: ProductRegionRecord[];
  union_rule: string;
  volume: number;
  log2_volume: number;
};

export type GenerationRequestOutcomeRecord = {
  kind: string;
  region?: StateSpaceRegionRecord;
  capacity_region?: StateSpaceRegionRecord;
  minimum_region?: StateSpaceRegionRecord;
};

export class StateSpaceRecordTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StateSpaceRecordTransportError';
  }
}

const stateSpaceError = (message: string) => new StateSpaceRecordTransportError(message);

export function parseStateSpaceRegionRecord(
  value: unknown,
  path = 'state_space_region',
  error = stateSpaceError,
): StateSpaceRegionRecord {
  const record = requireRecord(value, path, error);
  return {
    id: requireString(record.id, `${path}.id`, error),
    ambient: parseStateSpaceAmbientRecord(record.ambient, `${path}.ambient`, error),
    components: requireArray(record.components, `${path}.components`, error).map(
      (component, index) =>
        parseProductRegionRecord(component, `${path}.components.${index}`, error),
    ),
    union_rule: requireString(record.union_rule, `${path}.union_rule`, error),
    volume: requireNumber(record.volume, `${path}.volume`, error),
    log2_volume: requireNumber(record.log2_volume, `${path}.log2_volume`, error),
  };
}

export function parseGenerationRequestOutcomeRecord(
  value: unknown,
  path = 'request_outcome',
  error = stateSpaceError,
): GenerationRequestOutcomeRecord {
  const record = requireRecord(value, path, error);
  return {
    kind: requireString(record.kind, `${path}.kind`, error),
    region:
      record.region === undefined
        ? undefined
        : parseStateSpaceRegionRecord(record.region, `${path}.region`, error),
    capacity_region:
      record.capacity_region === undefined
        ? undefined
        : parseStateSpaceRegionRecord(record.capacity_region, `${path}.capacity_region`, error),
    minimum_region:
      record.minimum_region === undefined
        ? undefined
        : parseStateSpaceRegionRecord(record.minimum_region, `${path}.minimum_region`, error),
  };
}

function parseStateSpaceAmbientRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): StateSpaceAmbientRecord {
  const record = requireRecord(value, path, error);
  const fieldDomain = requireRecord(record.field_domain, `${path}.field_domain`, error);
  Object.entries(fieldDomain).forEach(([key, item]) => {
    if (key === '' || !['number', 'string'].includes(typeof item)) {
      throw error(`${path}.field_domain.${key}: expected scalar field-domain value`);
    }
  });
  return {
    field_domain_kind: requireString(record.field_domain_kind, `${path}.field_domain_kind`, error),
    field_domain: fieldDomain as Record<string, string | number>,
    field_codomain_id: requireString(record.field_codomain_id, `${path}.field_codomain_id`, error),
    distinguishability: parseDistinguishabilityRecord(
      record.distinguishability,
      `${path}.distinguishability`,
      error,
    ),
  };
}

function parseDistinguishabilityRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): DistinguishabilityRecord {
  const record = requireRecord(value, path, error);
  return {
    kind: requireString(record.kind, `${path}.kind`, error),
    metric_id: optionalString(record.metric_id, `${path}.metric_id`, error),
    resolution: optionalNumber(record.resolution, `${path}.resolution`, error),
    certificate_id: optionalString(record.certificate_id, `${path}.certificate_id`, error),
  };
}

function parseProductRegionRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): ProductRegionRecord {
  const record = requireRecord(value, path, error);
  return {
    axis_regions: requireArray(record.axis_regions, `${path}.axis_regions`, error).map(
      (axisRegion, index) =>
        parseAxisRegionRecord(axisRegion, `${path}.axis_regions.${index}`, error),
    ),
    measure_rule: requireString(record.measure_rule, `${path}.measure_rule`, error),
    volume: requireNumber(record.volume, `${path}.volume`, error),
    log2_volume: requireNumber(record.log2_volume, `${path}.log2_volume`, error),
    stratum_id: optionalString(record.stratum_id, `${path}.stratum_id`, error),
    stratum_target:
      record.stratum_target === undefined
        ? undefined
        : requireRecord(record.stratum_target, `${path}.stratum_target`, error),
  };
}

function parseAxisRegionRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): AxisRegionRecord {
  const record = requireRecord(value, path, error);
  return {
    axis: parseStateSpaceAxisRecord(record.axis, `${path}.axis`, error),
    coordinate_region: requireArray(record.coordinate_region, `${path}.coordinate_region`, error).map(
      (coordinate, index) => {
        if (typeof coordinate !== 'number' && typeof coordinate !== 'string') {
          throw error(`${path}.coordinate_region.${index}: expected coordinate scalar`);
        }
        return coordinate;
      },
    ),
    count: requireNumber(record.count, `${path}.count`, error),
    log2_count: requireNumber(record.log2_count, `${path}.log2_count`, error),
  };
}

function parseStateSpaceAxisRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): StateSpaceAxisRecord {
  const record = requireRecord(value, path, error);
  return {
    id: requireString(record.id, `${path}.id`, error),
    domain: parseAxisDomainRecord(record.domain, `${path}.domain`, error),
  };
}

function parseAxisDomainRecord(
  value: unknown,
  path: string,
  error = stateSpaceError,
): AxisDomainRecord {
  const record = requireRecord(value, path, error);
  const kind = requireString(record.kind, `${path}.kind`, error);
  if (kind === 'integer-range') {
    return {
      kind,
      lower: requireNumber(record.lower, `${path}.lower`, error),
      upper: requireNumber(record.upper, `${path}.upper`, error),
    };
  }
  if (kind === 'real-grid') {
    return {
      kind,
      lower: requireNumber(record.lower, `${path}.lower`, error),
      upper: requireNumber(record.upper, `${path}.upper`, error),
      count: requireNumber(record.count, `${path}.count`, error),
    };
  }
  if (kind === 'enumerated-cells') {
    return {
      kind,
      cells: requireArray(record.cells, `${path}.cells`, error).map((cell, index) =>
        requireString(cell, `${path}.cells.${index}`, error),
      ),
    };
  }
  if (kind === 'binary-vector') {
    return {
      kind,
      dimension: requireNumber(record.dimension, `${path}.dimension`, error),
    };
  }
  throw error(`${path}.kind is invalid`);
}

function optionalString(
  value: unknown,
  path: string,
  error = stateSpaceError,
): string | undefined {
  return value === undefined ? undefined : requireString(value, path, error);
}
