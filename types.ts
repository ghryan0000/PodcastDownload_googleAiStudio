export enum AppState {
  IDLE = 'IDLE',
  FETCHING_SOURCE = 'FETCHING_SOURCE',
  ANALYZING_CODE = 'ANALYZING_CODE',
  DOWNLOADING_ASSET = 'DOWNLOADING_ASSET',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export interface ExtractionResult {
  originalUrl: string;
  streamUrl: string | null;
  pageTitle?: string;
}

export interface StepStatus {
  id: string;
  label: string;
  status: 'pending' | 'loading' | 'completed' | 'error';
}