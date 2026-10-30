import { normalizeStepLimit, unsupportedPrefill } from './normalize.mjs';

export default {
  id: 'local.step-limit-compat',
  async setup(ctx) {
    await ctx.session.hook('context', (event) => {
      normalizeStepLimit(event.messages);
    }, { providerID: 'lmstudio' });
    // A changed upstream prompt must fail visibly, not enter deterministic retries.
    await ctx.session.hook('retry', (event) => {
      if (unsupportedPrefill(event.error)) event.decision = { retry: false };
    }, { providerID: 'lmstudio' });
  }
};
