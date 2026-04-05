#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "andrew.h"
#include "unicode.h"
#include "tagfile.h"
#include "lexicon.h"


entry *create_lexicon(const char *filename)
{
	char cont = YES;
	int i;

	unichar uniNULL[] =  { 0x004e, 0x0055, 0x004c, 0x004c, 0x0000 };

	FILE *source;

	entry *lexicon = NULL, *buffer, *current;

	token *word;
	
	/* open the source file and check for (then discard) directionality character */
	if (!(source = fopen(filename, "rb")))
	{
		error_file_open(filename);
		fcloseall();
		return NULL;
	}
	if (!( ucheckdir(source) ))
	{
		puts("Unicode file not recognising, lexicon creation aborts.");
		fcloseall();
		return NULL;
	}

	buffer = get_entry();

	while (word = load_token(source))
	{
		/* check for a NULL tag indicating SGML  element*/
		if ( ustrident(word->tag[0], uniNULL) )
			continue;

		/* copy relevant bits from token to buffer lexicon entry and jettison former */
		ustrcpy(buffer->wordform, word->wordform);
		ustrcpy(buffer->tag[0], word->tag[0]);
		buffer->freq[0] = 1;
		free(word);


		/* decapitalise the word from file */
		udecaps(buffer->wordform);
		// !!!! can't disemvowel words here, so perhaps a program to disemvowel on a file-scale?

		
		if (lexicon == NULL)
		{
			lexicon = buffer;
			buffer = get_entry();
		}
		else
		{
			if (!(current = find_entry(buffer->wordform, lexicon)))
			{
				/* if this entry is not on the list... */
				current = lexicon;
				while (current->next_entry)
					current = current->next_entry;
				current->next_entry = buffer;
				buffer = get_entry();
			}
			else	
			{
				for (i = 0; i < TAGSMAX; i++)
				{
					if (ustrident(buffer->tag[0], current->tag[i]))
					{
						/* if the tag is already in the taglist... */
						current->freq[i]++;
						break;
					}
					if (!current->tag[i][0])
					{
						/* if there is a slot free for a tag, the new tag goes in there */
						ustrcpy(current->tag[i], buffer->tag[0]);
						current->freq[i] = 1;
						break;
					}
				}
			}
		}
	}

	if(fclose(source))
	{
		error_file_close(filename);
		fcloseall();
	}

	rewrite_lexnumbers(lexicon);


	return lexicon;
}





/* note that this function doesn't even touch frequencies. They are ignored. */
/* the second lexicon is wiped from memory */

entry *merge_lexicon(entry *lex1, entry *lex2)
{
	unsigned char c1, c2;
	entry *lex3, *current1, *current2, *current3, *next2;


	lex3 = NULL;

	current1 = lex1;
	current2 = lex2;


	while (current2)
	{
		next2 = current2->next_entry;
		if (current1 = find_entry(current2->wordform, lex1))
		{
			/* patch tags onto entry */
			for (c2 = 0; c2 < TAGSMAX && current2->tag[c2][0] ; c2++)
			{
				for (c1 = 0; c1 < TAGSMAX; c1++)
				{
					if (ustrident(current2->tag[c2], current1->tag[c1]))
						break;
					if (!current1->tag[c1][0])
					{
						ustrcpy(current1->tag[c1], current2->tag[c2]);
						break;
					}
				}
			}
			free(current2);
		}
		else
		{
			if (!lex3)
				current3 = (lex3 = current2);
			else
				current3 = (current3->next_entry = current2);
			current2->next_entry = NULL;
		}
		current2 = next2;
	}

	/* join the first list and the thinned second, then sort */
	current1 = lex1;
	while (current1->next_entry)
		current1 = current1->next_entry;
	current1->next_entry = lex3;

	lex1 = tidy_lexicon(lex1);

	lex1 = sort_lexicon(lex1, NULL);

	rewrite_lexnumbers(lex1);

	return lex1;
	/* this return value should be reassigned to the argument */
}





entry *tidy_lexicon(entry *lex)
{
	int i, j;
	entry *current, *next, *prev, *finder;

	if (lex == NULL)
	{
		puts("Error! tidy_lexicon fed a NULL lexicon. Program aborts.");
		exit(1);
	}

	current = lex;
	prev = NULL;
	next = current->next_entry;

	while ( current )
	{
		finder = find_entry(current->wordform, lex);

		if ( finder != current )
		{
			/* there is a version of this word earlier up */
			/* patch tags of this entry onto that other entry */
			for (i = 0; i < TAGSMAX && current->tag[i][0] ; i++)
			{
				for (j = 0; j < TAGSMAX; j++)
				{
					if (ustrident(current->tag[i], finder->tag[j]))
						break;
					if ( ! finder->tag[j][0] )
					{
						ustrcpy(finder->tag[j], current->tag[i]);
						break;
					}
				}
			}

			/* remove current entry from the list */
			if ( current == lex )
				lex = next;
			else
				prev->next_entry = next;
			free(current);
			current = prev;

		}

		prev = current;
		current = next;
		if (current == NULL)
			break;
		next = current->next_entry;
	}

	return lex;
	/* this return value should be reassigned to the argument */
	/* in theory this is not necessary, but in practice, let's do it anyway */
}



/* returns 1 for save error, 0 for correct save */
int save_lexicon(entry *head, const char *filename, unsigned int threshold, char probs)
{
	char i, j;

	char c_buffer[50];

	unsigned long int probstotal = 0, calcprob;

	unichar car_ret[] = { 0x000d, 0x000a, 0x0000 };
	unichar space[]   = { 0x0020, 0x0000 };
	unichar tab[]     = { 0x0009, 0x0000 };
	unichar slash[]   = { 0x002f, 0x0000 };
	unichar aye[]     = { 0x0069, 0x0000 };

	unichar line[500];

	unichar u_buffer[50];
	
	FILE *dest;
	
	entry *current;

	if ( !(dest = fopen(filename, "wb")) )
	{
		error_file_open(filename);
		fcloseall();
		return 1;
	}
	if (fputuc(0xfeff, dest) == UERR)
	{
		error_file_write(filename);
		fcloseall();
		return 1;
	}
	
	for ( current = head ; current ; current = current->next_entry )
	{
		probstotal = 0;

		/* calculate total number of occurrences of token */
		for (i = 0; i < TAGSMAX && current->tag[i][0]; i++)
					probstotal += current->freq[i];

		/* if probstotal is doesn't reach threshold, then this entry won't be written */
		if ( ! (probstotal >= threshold) )
			continue;

		/* write "i" and idno and space to line buffer, then word followed by tab */
		ustrcpy(line, aye);
		ustrcat(line, current->idno);
		ustrcat(line, space);


		ustrcat(line, current->wordform);
		ustrcat(line, tab);


		/* now, for each tag write the tag, then if probs is true, write slash then a prob */
		for (i = 0; i < TAGSMAX && current->tag[i][0]; i++)
		{
			/* add a space, if this is not the first tag */
			if (i > 0)
				ustrcat(line, space);

			ustrcat(line, current->tag[i]);

			/* if probs is set to TRUE, then write for each tag its probability */
			if (probs)
			{
				ustrcat(line, slash);
				calcprob = (100 * current->freq[i]) / probstotal ;

				if (calcprob == 100)
					calcprob = 99;

				sprintf(c_buffer, "%d", calcprob );
				for ( j = 0 ; j < 20 ; j++ )
					if ( ! (u_buffer[j] = (unichar)c_buffer[j]) )
						break;
				ustrcat(line, u_buffer);
			}
		}
		ustrcat(line, car_ret);
		if (fputus(line, dest) == EOF)
		{
			error_file_write(filename);
			fcloseall();
			return 1;
		}
	}
	if (fclose(dest))
	{
		error_file_close(filename);
		fcloseall();
		return 1;
	}
	return 0;
}




/* returns NULL if there is an error */
entry *load_lexicon(const char *filename)
{
	unsigned char x, y;
	char done = NO, done_within = NO, thistagprobbed = NO;
	
	unichar junk;
	unichar s_junk[500];

	entry *lexicon = NULL, *current, *buffer;
	FILE *source;

	/* open the source file and check for (then discard) directionality character */
	if (!(source = fopen(filename, "rb")))
	{
		error_file_open(filename);
		fcloseall();
		return NULL;
	}
	if (!( ucheckdir(source) ))
	{
		fputs("Specified lexicon source file not recognised as Unicode!", stderr);
		fcloseall();
		return NULL;
	}
	
	while (done == NO)
	{
		buffer = get_entry();

		/* first, skip 1 character, and skip comment lines */
		if ((junk = fgetuc(source)) == UERR)
		{
			free(buffer);
			break;
		}
		if (junk != 0x0069)
		{
			free(buffer);
			if (fgetus(s_junk, source) == NULL)
				break;
			else
				continue;
		}
		/* now, load the idno */
		for (x = 0; x < IDNOLENGTH; x++)
		{
			if((buffer->idno[x] = fgetuc(source)) == UERR)
			{
				done = YES;
				buffer->idno[x] = 0;
				break;
			}
			if(buffer->idno[x] == 0x0020)
			{
				buffer->idno[x] = 0;
				break;
			}
		}
		/* and null-terminate */
		buffer->idno[IDNOLENGTH-1] = 0x0000;

		/* load the word */
		for (x = 0; x < WORDLENGTH; x++)
		{
			if((buffer->wordform[x] = fgetuc(source)) == UERR)
			{
				done = YES;
				buffer->wordform[x] = 0;
				break;
			}
			if(buffer->wordform[x] == 0x0009)
			{
				buffer->wordform[x] = 0;
				break;
			}
		}
		/* ensure the string is null-terminated */
		buffer->wordform[WORDLENGTH-1] = 0x0000;

		done_within = NO;
		for (y = 0; y < TAGSMAX && (done == NO && done_within == NO); y++)
		{
			thistagprobbed = NO;
			for (x = 0; x < TAGLENGTH && (done == NO && done_within == NO); x++)
			{
				/* if the end of the file is reached */
				if ((buffer->tag[y][x] = fgetuc(source)) == UERR)
				{
					done = YES;
					buffer->tag[y][x] = 0;
					break;
				}
				/* assorted checks on things that could signal end of a tag */
				if (buffer->tag[y][x] == 0x0020)
				{
					buffer->tag[y][x] = 0;
					break;
				}
				if (buffer->tag[y][x] == 0x000d)
				{
					done_within = YES;
					buffer->tag[y][x] = 0;
					break;
				}
				if (buffer->tag[y][x] == 0x002f)
				{
					thistagprobbed = YES;
					buffer->tag[y][x] = 0;
					break;
				}
			}
			buffer->tag[y][WORDLENGTH-1] = 0;

			if (thistagprobbed)
			{
				/* load two chars, change characters to digits, and create frequency score */
				if ((s_junk[0] = fgetuc(source)) == UERR)
				{
					done = YES;
					break;
				}
				if ((s_junk[1] = fgetuc(source)) == UERR)
				{
					done = YES;
					break;
				}

				s_junk[0] -= 0x0030;
				s_junk[1] -= 0x0030;

				buffer->freq[y] = (10 * s_junk[0]) + s_junk[1];
				s_junk[0] = 0x0000;

				/* load another char. If 000d, then move on out. */
				if ((junk = fgetuc(source)) == UERR)
				{
					done = YES;
					break;
				}
				if (junk == 0x000d)
					done_within = YES;
			}
		}

		if (buffer->wordform[0])
		{
			if (!lexicon)
				current = (lexicon = buffer);
			else
				current = (current->next_entry = buffer);
		}
		else
			free(buffer);

		/* get to start of next line) */
		while (junk != 0x000a)
			if ((junk = fgetuc(source)) == UERR)
			{
				done = YES;
				break;
			}
		junk = 0;
	}

	if(fclose(source))
	{
		error_file_close(filename);
		fcloseall();
	}
	return lexicon;
}




entry *sort_lexicon(entry *lexicon, int (*sort_entry)(entry *first, entry *second) )
/* second argument: pointer to the function that sorts the 
/* returns a pointer to the sorted lexicon (will usually be reassigned to argument) */
{
	char sorted = NO;
	entry *current, *previous, *next;

	if (!sort_entry)
		sort_entry = sort_entry_wordform;
	
	while (lexicon && sorted == NO)
	{
		for (current = lexicon, next = lexicon->next_entry, sorted = YES ; next ; )
		{
			while (sort_entry(current, next))
			{
				previous = current;
				current = next;
				if (!(next = next->next_entry))
					break;
			}
			if(!next)
				break;

			if (current == lexicon)
				lexicon = next;
			else
				previous->next_entry = next;

			current->next_entry = next->next_entry;
			next->next_entry = current;

			previous = current;
			current = next;
			next = next->next_entry;

			sorted = NO;
		}
	}
	return lexicon;
}



void rewrite_lexnumbers(entry *lexicon)
{
	int i, j;

	entry *current;

	char buffer[IDNOLENGTH];

	current = lexicon;

	for ( i = 1 ; current ; current = current->next_entry , i++ )
	{
		sprintf(buffer, "%06d", i);

		for ( j = 0 ; j < IDNOLENGTH ; j++ )
			if ( (current->idno[j] = (unichar)buffer[j]) == 0x0000 )
				break;
	}
}



void free_lexicon(entry *head)
{
	entry *current = head;
	entry *next;

	while (current)
	{
		next = current->next_entry;
		free(current);
		current = next;
	}
}



void blank_lexicon(const char *filename, unsigned int lines)
{
	long unsigned int i;
	int j;
	char buffer[20];
	unichar number[20];
	FILE *dest;

	unichar endline[] = { 0x0020, 0x0009, 0x000d, 0x000a, 0x0000 };
	
	/* open file to write */
	if( !(dest = fopen(filename, "wb")) )
	{
		puts("Error opening blank lexicon file!");
		fcloseall();
		return;
	}

	/* insert directionality character */
	if ( fputuc( RIGHTWAY , dest) == UERR )
	{
		puts("Error writing to blank lexicon file!");
		fcloseall();
		return;
	}

	/* write the lines to file */
	for ( i = 1 ; i <= lines ; i++ )
	{
		sprintf(buffer, "i%06d", i);

		for ( j = 0 ; j < 20 ; j++ )
			if ( ! (number[j] = (unichar)buffer[j]) )
				break;

		ustrcat(number, endline);

		if (fputus(number, dest) == EOF)
		{
			puts("Error writing to file!");
			fcloseall();
			return;
		}

	}

	if (fclose(dest) < 0)
	{
		puts("Error closing finished blank lexicon file!");
		fcloseall();
		return;
	}
}



entry *get_entry(void)
{
	unsigned char x;
	entry *pointer;
	if(!(pointer = (entry *)malloc(sizeof(entry))))
	{
		error_memory_alloc();
		fcloseall();
		exit(1);
	}
	pointer->next_entry = NULL;
	for(x = 0; x < TAGSMAX; x++)
		pointer->tag[x][0] = 0;
	pointer->wordform[0] = 0;
	pointer->idno[0] = 0;
	return pointer;
}




/* returns pointer to place on list where word is found */
/* returns NULL if it's not found */
entry *find_entry(const unichar *target, entry *head)
{
	entry *current;

	if (!(current = head))
	{
		puts("find_entry was fed a null pointer, aborting program");
		exit(1);
	}

	while ( ! (ustrident(target, current->wordform)) )
		if ((current = current->next_entry) == NULL)
			break;

	return current;
}


void assign_tags(token *dest, entry *source)
{
	int i;

	for ( i = 0 ;   i < TAGSMAX  &&  source->tag[i][0]   ; i ++)
		ustrcpy(dest->tag[i], source->tag[i]);
}




int sort_entry_wordform(entry *first, entry *second)
/* returns 1 if first comes before second in the alphabet, 0 if second comes		*/
/* before first, and EOF if they are the same. Returns UERR for error.										*/
{
	unichar *u_first = first->wordform;
	unichar *u_second = second->wordform;

	while (*u_first == *u_second )
	{
		if (!(*u_first))
			return EOF;
		u_first++, u_second++;
	}
	if (*u_first < *u_second)
		return 1;
		/* End-of-string (NULL) is considered the first letter in all the alphabets	*/
	if (*u_first > *u_second)
		return 0;
	return UERR;
}


int sort_entry_tag(entry *first, entry *second)
/* returns 1 if first comes before second in the alphabet, 0 if second comes		*/
/* before first, and EOF if they are the same. Returns UERR for error.										*/
{
	unichar *u_first = first->tag[0];
	unichar *u_second = second->tag[0];

	while (*u_first == *u_second )
	{
		if (!(*u_first))
			return EOF;
		u_first++, u_second++;
	}
	if (*u_first < *u_second)
		return 1;
		/* End-of-string (NULL) is considered the first letter in all the alphabets	*/
	if (*u_first > *u_second)
		return 0;
	return UERR;
}



/* returns YES or NO */
int entry_tagsinc(entry *item, const unichar *thistag)
{
	int i;

	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( ustrident(item->tag[i], thistag) )
			return YES;
	}

	return NO;
}

void entry_addtag(entry *item, const unichar *thistag)
{
	int i;

	if ( entry_tagsinc(item, thistag) )
		return;

	for ( i = 0 ; i < TAGSMAX ; i++ )
	{
		if ( item->tag[i][0] == 0 )
		{
			ustrcpy(item->tag[i], thistag);

			if ( i != TAGSMAX-1 )
				item->tag[i+1][0] = 0;

			break;
		}
	}
}