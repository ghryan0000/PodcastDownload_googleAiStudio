import React from 'react';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';
import { StepStatus } from '../types';

interface StepIndicatorProps {
  steps: StepStatus[];
}

const StepIndicator: React.FC<StepIndicatorProps> = ({ steps }) => {
  return (
    <div className="w-full max-w-2xl mx-auto my-8 space-y-3">
      {steps.map((step, index) => {
        let icon = <Circle className="w-5 h-5 text-gray-300" />;
        let textColor = 'text-gray-400';
        let borderColor = 'border-gray-100';
        let bgColor = 'bg-white';
        let shadow = '';

        if (step.status === 'loading') {
          icon = <Loader2 className="w-5 h-5 text-rose-500 animate-spin" />;
          textColor = 'text-rose-600 font-semibold';
          borderColor = 'border-rose-100';
          bgColor = 'bg-rose-50';
          shadow = 'shadow-sm';
        } else if (step.status === 'completed') {
          icon = <CheckCircle2 className="w-5 h-5 text-green-500" />;
          textColor = 'text-gray-900 font-medium';
          borderColor = 'border-gray-200';
          bgColor = 'bg-white';
          shadow = 'shadow-sm';
        } else if (step.status === 'error') {
          icon = <XCircle className="w-5 h-5 text-red-500" />;
          textColor = 'text-red-600 font-medium';
          borderColor = 'border-red-100';
          bgColor = 'bg-red-50';
          shadow = 'shadow-sm';
        }

        return (
          <div 
            key={step.id} 
            className={`flex items-center p-4 rounded-xl border ${borderColor} ${bgColor} ${shadow} transition-all duration-300`}
          >
            <div className="mr-4">
              {icon}
            </div>
            <span className={`text-sm ${textColor}`}>
              {step.label}
            </span>
            {step.status === 'completed' && (
              <span className="ml-auto text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full font-semibold">DONE</span>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default StepIndicator;