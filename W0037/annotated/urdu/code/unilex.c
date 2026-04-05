#include <stdio.h>
#include "andrew.h"
#include "unicode.h"
#include "tagfile.h"
#include "lexicon.h"



main(int argc, char *argv[])
{

	char filename[260];

	unsigned short int templatelength;

	unsigned short int i;

	char probs;

	entry *lex;
	entry *lex_2;


	puts(" ");

	if (argc != 1)
	{
		/* potential here for Unilex to be called with an arg  */
		/* nothing written for this yet; I did it all manually */
		/* so this path drops straight to return 0 at the end  */
	}
	else
	{
		puts("\nWelcome to Unilex!");
		while (1)
		{
			fflush(stdin);

			puts("\n\n-----------------------------------------\n");

			puts("\n\nChoose an option:\n");
			puts("[B]\tcreate a blank lexicon template");
			puts("[D]\tderive lexicon from file");
			puts("[S]\tsort a lexicon");
			puts("[M]\tmerge two lexicons");
			puts("[T]\tsort a lexicon by tag\n");
			
			puts("[X]\texit program\n");

			switch(input_one_touch("BDSMTX"))
			{

			case 'D':

				puts("\n\nSpecify file to derive lexicon from, or [ENTER] to cancel:\n");
				input_string(filename);

				if (filename[0] == 0)
					break;

				/* run the function! */
				if ( ! (lex = create_lexicon(filename)) )
					break;

				puts("\n\nSpecify filename to save lexicon under, or [ENTER] to cancel:\n");
				input_string(filename);

				if ( filename[0] != 0 )
				{
					puts("\n\nDo you want to save the lexicon with probabilities?\n");
					probs = input_yes_no();

					puts("\n\nSet a frequency threshold. Wordforms must occur at least this many");
					puts("times to be saved in the lexicon. To include every word, enter zero.\n");
					i = input_integer();

					save_lexicon(lex, filename, i, probs);
					
					puts("\nDone!\n");
				}
				else
					puts("Save cancelled!");

				free_lexicon(lex);

				break;

			case 'M':

				/* acquire lex name and load it */
				puts("\n\nSpecify filename of first lexicon, or [ENTER] to cancel:");
				input_string(filename);

				if (filename[0] == 0)
					break;

				if ( ! (lex = load_lexicon(filename)) )
				{
					puts("Error loading lexicon.");
					break;
				}

				/* acquire second lex name and load it */
				puts("\n\nSpecify filename of second lexicon, or [ENTER] to cancel:");
				input_string(filename);

				if (filename[0] == 0)
					break;

				if ( ! (lex_2 = load_lexicon(filename)) )
				{
					puts("Error loading lexicon.");
					break;
				}

				/* call the combining function */
				lex = merge_lexicon(lex, lex_2);

				/* save it */
				puts("\n\nSpecify filename to save merged lexicon under, or [ENTER] to cancel:\n");
				input_string(filename);

				if ( filename[0] != 0 )
				{
					puts("\nMerged lexicons are saved without probabilities. All words are saved.");

					save_lexicon(lex, filename, 0, NO);
					
					puts("\nDone!\n");
				}
				else
					puts("Save cancelled!");

				free_lexicon(lex);
				
				break;


			case 'B':

				puts("\n\nEnter a name for the file you wish to create, or [ENTER] to cancel:");
				input_string(filename);

				if (filename[0] == 0)
					break;

				puts("\n\nHow many empty, numbered lines would you like this blank file to contain?");

				/* get a number from them and assign it to templatelength */
				scanf("%d", &templatelength);

				blank_lexicon(filename, templatelength);

				break;


			case 'S':

				puts("\n\nSpecify filename of lexicon to sort, or [ENTER] to cancel:\n");
				input_string(filename);

				if (filename[0] == 0)
					break;

				if ( ! (lex = load_lexicon(filename)) )
				{
					puts("Error loading lexicon.");
					break;
				}

				lex = sort_lexicon(lex, NULL);

				puts("\n\nEnter filename for sorted lexicon, or [ENTER] to cancel:\n");
				input_string(filename);

				if ( (filename[0] != 0) )
				{
					puts("\n\nDo you want to save the lexicon with probabilities?\n");
					probs = input_yes_no();
					save_lexicon(lex, filename, 0, probs);
					puts("\nDone!");
				}

				free_lexicon(lex);

				break;


			case 'T':

				puts("\n\nSpecify filename of lexicon to sort, or [ENTER] to cancel:\n");
				input_string(filename);

				if (filename[0] == 0)
					break;

				if ( ! (lex = load_lexicon(filename)) )
				{
					puts("Error loading lexicon.");
					break;
				}

				lex = sort_lexicon(lex, sort_entry_tag);

				puts("\n\nEnter filename for sorted lexicon, or [ENTER] to cancel:\n");
				input_string(filename);

				if ( (filename[0] != 0) )
				{
					puts("\n\nDo you want to save the lexicon with probabilities?\n");
					probs = input_yes_no();
					save_lexicon(lex, filename, 0, probs);
					puts("\nDone!");
				}

				free_lexicon(lex);

				break;


			case 'X':
				puts(" ");
				return 0;

			default:
				break;
			}
		}
	}
	puts(" ");
	return 0;
}