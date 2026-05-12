use std::sync::Arc;

use crate::{Result, TurError};
use llguidance::ParserFactory as LlgParserFactory;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::ByteTokenizer;
use tracing::warn;

pub type ParserFactory = LlgParserFactory;

pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
) -> Result<Arc<ParserFactory>> {
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    let target_vocab = vocab_size.map(|v| {
        if v < tokenizer_vocab {
            warn!(
                "Requested vocab size {} is smaller than tokenizer vocab size {}. Using tokenizer size.",
                v,
                tokenizer_vocab
            );
            tokenizer_vocab
        } else {
            v
        }
    });
    let env = ByteTokenizer::from_tokenizer(tokenizer)
        .map_err(|e| TurError::Guidance(e.to_string()))?
        .into_tok_env(target_vocab)
        .map_err(|e| TurError::Guidance(e.to_string()))?;
    let factory = ParserFactory::new_simple(&env).map_err(|e| TurError::Guidance(e.to_string()))?;
    Ok(Arc::new(factory))
}
