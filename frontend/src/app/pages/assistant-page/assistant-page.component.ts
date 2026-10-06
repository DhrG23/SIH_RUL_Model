import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AssistantChatService, ChatTurn } from '../../core/services/assistant-chat.service';

@Component({
  selector: 'app-assistant-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './assistant-page.component.html',
  styleUrls: ['./assistant-page.component.css'],
})
export class AssistantPageComponent {
  private readonly assistant = inject(AssistantChatService);

  readonly messages = signal<ChatTurn[]>([
    { role: 'assistant', text: 'Ask me about the current engine telemetry, diagnostics, or mission state.' },
  ]);
  readonly draft = signal('');
  readonly isSending = signal(false);
  readonly error = signal('');
  readonly modelUsed = signal('');

  async send(): Promise<void> {
    const question = this.draft().trim();
    if (!question || this.isSending()) return;

    // Snapshot the prior turns BEFORE appending the new question, so the
    // server gets "everything said so far" and "the new question" separately.
    const priorHistory = this.messages();

    this.messages.update(messages => [...messages, { role: 'operator', text: question }]);
    this.draft.set('');
    this.error.set('');
    this.isSending.set(true);
    try {
      const { reply, model } = await this.assistant.ask(question, priorHistory);
      this.messages.update(messages => [...messages, { role: 'assistant', text: reply }]);
      this.modelUsed.set(model);
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
