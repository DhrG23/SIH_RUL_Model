import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface ChatTurn {
  role: 'operator' | 'assistant';
  text: string;
}

export interface AssistantChatResponse {
  reply: string;
  model: string;
}

/** Talks to the telemetry server's Gemini-backed engine assistant. */
@Injectable({ providedIn: 'root' })
export class AssistantChatService {
  private readonly http = inject(HttpClient);

  /** Sends the new question plus prior turns so follow-ups keep their context. */
  ask(message: string, history: ChatTurn[] = []): Promise<AssistantChatResponse> {
    return firstValueFrom(
      this.http.post<AssistantChatResponse>(`${environment.httpUrl}/api/assistant-chat`, { message, history }),
    );
  }
}
