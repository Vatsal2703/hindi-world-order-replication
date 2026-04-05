void urdutag_file(const char *input_filename, const char *output_filename,
				  const char *lexicon_filename);

void urdutag(entry *lexicon, token *word);

void urduanalyse(token *word, unichar tagset[TAGSETSIZE][TAGLENGTH]);

token *do_the_splits(token *word, entry *lexicon, FILE *dest);

void disemvowel(unichar *string);