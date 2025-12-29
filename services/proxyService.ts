/**
 * Helper function to try multiple proxy services in sequence.
 * This provides redundancy if one proxy is down or blocked.
 */
async function fetchWithFallbacks(url: string, asBlob: boolean = false): Promise<string | Blob> {
  // List of proxies to try.
  // 1. AllOrigins: Reliable, JSON/Raw support.
  // 2. CorsProxy.io: Fast, supports many headers.
  // 3. CodeTabs: Good fallback.
  // 4. ThingProxy: Another option.
  const proxyGenerators = [
    (u: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
    (u: string) => `https://corsproxy.io/?${encodeURIComponent(u)}`,
    (u: string) => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(u)}`,
    (u: string) => `https://thingproxy.freeboard.io/fetch/${u}`
  ];

  let lastError: unknown = null;

  for (const generateUrl of proxyGenerators) {
    try {
      const proxyUrl = generateUrl(url);
      const response = await fetch(proxyUrl);
      
      if (!response.ok) {
        // If the proxy returns an error (404/500), it might be the proxy or the target.
        // We try the next proxy just in case.
        throw new Error(`Proxy responded with status ${response.status}`);
      }

      if (asBlob) {
        return await response.blob();
      } else {
        const text = await response.text();
        // Simple validation to ensure we didn't get an empty response
        if (!text || text.trim().length === 0) {
          throw new Error("Received empty response");
        }
        return text;
      }
    } catch (err) {
      console.warn(`Proxy attempt failed for ${url} using generator`, err);
      lastError = err;
      // Continue to next proxy
    }
  }

  // If we exhaust all proxies
  throw new Error(
    lastError instanceof Error 
      ? `Failed to fetch via proxies: ${lastError.message}` 
      : "All proxy services failed to retrieve the content."
  );
}

/**
 * Fetches the raw HTML content of a URL using a rotation of CORS proxies.
 */
export async function fetchUrlSource(url: string): Promise<string> {
  try {
    const result = await fetchWithFallbacks(url, false);
    return result as string;
  } catch (error) {
    console.error("Fetch Source Error:", error);
    throw new Error("Unable to load the website. It may be blocking access or the URL is invalid.");
  }
}

/**
 * Downloads a file from a URL, handling CORS via proxy if necessary.
 */
export async function downloadFile(url: string, filename: string): Promise<void> {
  try {
    let blob: Blob;

    // 1. Try direct fetch (works if the CDN has CORS enabled)
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Direct fetch failed");
      blob = await response.blob();
    } catch (directError) {
      console.log("Direct download failed, attempting via proxies...", directError);
      // 2. Fallback to proxies
      blob = (await fetchWithFallbacks(url, true)) as Blob;
    }

    // Create download link
    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename.endsWith('.mp3') ? filename : `${filename}.mp3`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(objectUrl);
    
  } catch (error) {
    console.error("Download Error:", error);
    throw new Error("Could not download the file. It might be protected or unavailable.");
  }
}

/**
 * Fetches audio data and returns it as a Base64 string for API consumption.
 */
export async function fetchAudioAsBase64(url: string): Promise<string> {
  try {
    let blob: Blob;
    // Try direct first, then proxies
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("Direct fetch failed");
        blob = await response.blob();
    } catch {
        blob = await fetchWithFallbacks(url, true) as Blob;
    }

    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:audio/mp3;base64,")
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error("Audio Base64 Fetch Error:", error);
    throw new Error("Could not retrieve audio data for transcription.");
  }
}