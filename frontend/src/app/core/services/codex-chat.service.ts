import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../../environments/environment';

interface CodexChatResponse {
  reply: string;
}

@Injectable({ providedIn: 'root' })
export class CodexChatService {
  private readonly http = inject(HttpClient);

  ask(message: string): Promise<string> {
    return firstValueFrom(
      this.http.post<CodexChatResponse>(`${environment.httpUrl}/api/codex-chat`, { message }),
    ).then(response => response.reply);
  }
}
