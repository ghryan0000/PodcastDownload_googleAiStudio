import { GoogleGenAI, Type } from "@google/genai";

const getAI = () => {
  const key = process.env.API_KEY || process.env.GEMINI_API_KEY || "";
  return new GoogleGenAI({ apiKey: key });
};

/**
 * Uses Gemini to analyze HTML source code and locate the 'streamURL'.
 * We rely on Gemini's understanding of code syntax (JS/JSON objects) to find the key.
 */
export async function extractStreamUrlFromSource(sourceCode: string): Promise<string | null> {
  // Truncate extremely large HTML files to avoid unnecessary token usage, 
  // though Gemini 1.5/2.0 context is huge, 100k chars is usually enough for the head/body scripts where vars live.
  // Using a larger buffer to be safe.
  const truncatedSource = sourceCode.length > 500000 ? sourceCode.substring(0, 500000) : sourceCode;

  try {
    const response = await getAI().models.generateContent({
      model: 'gemini-1.5-flash',
      contents: `
        You are a code extraction expert. 
        I have provided the HTML source code of a webpage below.
        
        TASK:
        1. Scan the code for a variable, JSON key, or assignment named "streamURL".
        2. It typically looks like: "streamURL": "http...", var streamURL = '...', or streamURL: '...'.
        3. Extract the full URL value associated with it.
        4. If "streamURL" is not found, look for any other variable that obviously contains an MP3 audio stream URL (ending in .mp3 or typical streaming format).
        5. Return the result in a strict JSON format.

        SOURCE CODE:
        \`\`\`html
        ${truncatedSource}
        \`\`\`
      `,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            found: { type: Type.BOOLEAN },
            url: { type: Type.STRING, description: "The extracted stream URL" },
            confidence: { type: Type.STRING, description: "Low, Medium, or High confidence" }
          },
          required: ["found"]
        }
      }
    });

    const result = JSON.parse(response.text || '{}');

    if (result.found && result.url) {
      return result.url;
    }

    return null;

  } catch (error) {
    console.error("Gemini Extraction Error:", error);
    throw new Error("AI analysis failed.");
  }
}

/**
 * Transcribes audio content to text using Gemini.
 */
export async function transcribeAudio(base64Audio: string): Promise<string> {
  try {
    const response = await getAI().models.generateContent({
      model: 'gemini-2.0-flash-exp',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'audio/mp3',
              data: base64Audio
            }
          },
          {
            text: `
            Transcribe this audio file into high-quality text.
            
            Formatting Rules:
            1. Segment the text into clear, logical paragraphs based on the flow of speech.
            2. If there are distinct speakers, separate their turns with new lines.
            3. Use proper punctuation, capitalization, and sentence structure.
            4. Do not include timestamps or meta-commentary (like [music playing]).
            5. Output plain text with double newlines between paragraphs.
            `
          }
        ]
      }
    });

    return response.text || "No transcription available.";
  } catch (error) {
    console.error("Transcription Error:", error);
    throw new Error("AI transcription failed.");
  }
}