import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CodexChatService } from '../../core/services/codex-chat.service';

interface ChatMessage {
  role: 'operator' | 'assistant';
  text: string;
}

@Component({
  selector: 'app-codex-chat-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './codex-chat-page.component.html',
  styleUrls: ['./codex-chat-page.component.css'],
})
export class CodexChatPageComponent {
  private readonly codex = inject(CodexChatService);

  readonly messages = signal<ChatMessage[]>([
    { role: 'assistant', text: 'I can analyze the current engine telemetry, diagnostics, and mission state.' },
  ]);
  readonly draft = signal('');
  readonly isSending = signal(false);
  readonly error = signal('');

  async send(): Promise<void> {
    const question = this.draft().trim();
    if (!question || this.isSending()) return;

    this.messages.update(messages => [...messages, { role: 'operator', text: question }]);
    this.draft.set('');
    this.error.set('');
    this.isSending.set(true);
    try {
      const reply = await this.codex.ask(question);
      this.messages.update(messages => [...messages, { role: 'assistant', text: reply }]);
    } catch (error: unknown) {
      const detail = error && typeof error === 'object' && 'error' in error
        ? (error as { error?: { detail?: string } }).error?.detail
        : undefined;
      this.error.set(detail || 'The assistant could not answer this question.');
    } finally {
      this.isSending.set(false);
    }
  }
}
