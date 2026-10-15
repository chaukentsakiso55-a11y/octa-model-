export type Role = "system" | "user" | "assistant";
export type Message = { role: Role; content: string };
export type ReasonMode = "standard" | "deep";
export type ResearchSource = { id: number; title: string; url: string; snippet: string };
export type ReasonResult = { answer: string; mode: ReasonMode; passes: number; sources: ResearchSource[]; verified: boolean };

export class OctaClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly fetcher: typeof fetch = fetch
  ) {
    if (!baseUrl.startsWith("https://") && !/^http:\/\/localhost(?::\d+)?\/?$/.test(baseUrl)) {
      throw new Error("Octa API requires HTTPS outside localhost");
    }
    if (!apiKey) throw new Error("Octa API key is required");
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await this.fetcher(this.baseUrl.replace(/\/$/, "") + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`Octa API error: ${response.status}`);
    return response.json() as Promise<T>;
  }

  chat(messages: Message[], max_tokens = 512): Promise<unknown> {
    return this.post("/v1/chat/completions", { messages, max_tokens });
  }

  reason(task: string, mode: ReasonMode = "standard", use_web = false): Promise<ReasonResult> {
    return this.post("/v1/reason", { task, mode, use_web });
  }

  research(question: string): Promise<{ answer: string; sources: ResearchSource[]; verified: boolean }> {
    return this.post("/v1/research/answer", { question });
  }
}
