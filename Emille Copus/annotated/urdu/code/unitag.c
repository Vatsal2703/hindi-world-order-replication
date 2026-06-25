#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "andrew.h"
#include "commandline.h"

#define SIZE_FILENAME 500


main(int argc, char *argv[])
{
	int i, j;

	char input_filename[SIZE_FILENAME];
	char output_filename[SIZE_FILENAME+8];

	char *tokenised_filename = "__tokenised_utg.txt";
	char *analysed_filename = "__analysed_utg.txt";
	char *disambiguated_filename = "__disambiguated_utg.txt";
	char *decided_filename = "__decided_utg.txt";

	char instantiation[8][SIZE_FILENAME];

#define analyser instantiation[0]
#define disambiguator instantiation[2]
#define decider instantiation[4]
#define improver instantiation[6]

#define arg_analyser instantiation[1]
#define arg_disambiguator instantiation[3]
#define arg_decider instantiation[5]
#define arg_improver instantiation[7]


	char *nullstring = "NULL";

	char callbuffer[600];

	char *listsource_filename;

	FILE *test;
	FILE *instsource;
	FILE *listsource;



	/* handle the arguments */
	switch (argc)
	{
	case 3:
		listsource_filename = "__tempfilelist_utg.txt";

		/* open listsource, write listsourcefilename, close */
		if ( ! (listsource = fopen(listsource_filename, "w")) )
		{
			printf("The instantiation file \"%s\" does not exist or cannot be opened.\n", argv[1]);
			puts("Unitag aborts.");
			fcloseall();
			return 1;
		}
		if (fputs(argv[2], listsource) == EOF)
		{
			puts("Error writing to internal file. Unitag aborts.");
			fcloseall();
			return 1;
		}
		if (fclose(listsource) != 0)
		{
			cl_error_file_close(listsource_filename);
			fcloseall();
			return 1;
		}
		break;

	case 4:
		if (strident(argv[3], "L") || strident(argv[3], "l") )
			listsource_filename = argv[3];

		else
		{
			puts("Incorrect third argument. Correct call format:");
			puts(" ");
			puts("unitag instantiation_filename raw_text_filename");
			puts("OR");
			puts("unitag instantiation_filename listfile_filename L");
			puts(" ");
			return 1;
		}
		break;

	default:
		puts("Incorrect number of arguments. Correct call format:");
		puts(" ");
		puts("unitag instantiation_filename raw_text_filename");
		puts("OR");
		puts("unitag instantiation_filename listfile_filename L");
		puts(" ");
		return 1;
	}




	/* open instantiation file and set appropriate variables */
	if ( ! (instsource = fopen(argv[1], "r")) )
	{
		printf("The instantiation file \"%s\" does not exist or cannot be opened.\n", argv[1]);
		puts("Unitag aborts.");
		fcloseall();
		return 1;
	}

	/* load lines */
	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( ! fgets(instantiation[i], SIZE_FILENAME, instsource) )
		{
			puts("There was an error reading the instantiation file.");
			puts("Check that this file has all necessary details.");
			return 1;
		}

		/* trim off the carriage return at the end of the line, if necessary */
		if (instantiation[i][j = (strlen(instantiation[i])-1)] == '\n')
			instantiation[i][j] = 0;
		/* pass over comments and empty lines */
		if ( instantiation[i][0] == 0x2f || instantiation[i][0] == 0 )
		{
			i--;
			continue;
		}

	}

	/* remove specifiers */
	strcpy(callbuffer, &analyser[9]);
	strcpy(analyser, callbuffer);
	strcpy(callbuffer, &disambiguator[14]);
	strcpy(disambiguator, callbuffer);
	strcpy(callbuffer, &decider[8]);
	strcpy(decider, callbuffer);
	strcpy(callbuffer, &improver[9]);
	strcpy(improver, callbuffer);

	/* close instantiation file */
	if (fclose(instsource) != 0)
	{
		cl_error_file_close(argv[1]);
		fcloseall();
		return 1;
	}

	/* debug code, now commented out
	for ( i = 0 ; i < 8 ; i++ ) printf("%s\n", instantiation[i]);	*/



	/* open filename source list for ASCVII read */
	if ( ! (listsource = fopen(listsource_filename, "r") ) )
	{
		printf("The source file \"%s\" does not exist or cannot be opened.\n", listsource_filename);
		puts("Unitag aborts.");
		fcloseall();
		return 1;
	}



	while (fgets(input_filename, SIZE_FILENAME, listsource))
	{
		/* trim off the carriage return at the end */
		if (input_filename[i = (strlen(input_filename)-1)] == '\n')
			input_filename[i] = 0;

		/* is there a filename there? */
		if (!input_filename[0])
			continue;

		/* create an output filename */
		strcpy(output_filename, input_filename);
		strcat(output_filename, "_utg.txt");

		/* does the file in the buffer exist ? */
		if ( !(test = fopen(input_filename, "rb")) )
		{
			printf("The input file \"%s\" does not exist or cannot be opened.\n", input_filename);
			puts("It has not been tagged.");
			continue;
		}
		else if (fclose(test) != 0)
		{
			fputs("A test file has failed to close -- program aborting.", stderr);
			fcloseall();
			return 1;
		}


		/* run verticalise */
		sprintf(callbuffer, "verticalise \"%s\" \"%s\" *UT", input_filename, tokenised_filename);

		if (system(callbuffer))
		{
			printf("Error reported by verticalise for file %s!\n", input_filename);
			printf("Unitag aborting tagging run for this file.");
			continue;
		}


		/* run the specified analyser */
		sprintf(callbuffer, "%s \"%s\" \"%s\"", analyser, tokenised_filename, analysed_filename);
		if ( ! strident(arg_analyser, nullstring) )
		{
			strcat(callbuffer, " ");
			strcat(callbuffer, arg_analyser);
		}

		if (system(callbuffer))
		{
			printf("Error reported by %s for file %s!\n", analyser, input_filename);
			printf("Unitag aborting tagging run for this file.");
			continue;
		}


		/* run the specified disambiguator, if any */
		if ( strident(disambiguator, nullstring) )
			rename(analysed_filename, disambiguated_filename);

		else
		{
			sprintf(callbuffer, "%s \"%s\" \"%s\"", disambiguator, analysed_filename, disambiguated_filename);
			if ( ! strident(arg_disambiguator, nullstring) )
			{
				strcat(callbuffer, " ");
				strcat(callbuffer, arg_disambiguator);
			}

			if (system(callbuffer))
			{
				printf("Error reported by %s for file %s!\n", disambiguator, input_filename);
				printf("Unitag aborting tagging run for this file.");
				continue;
			}
		}

		/* run the specified decider, if any */
		if ( strident(decider, nullstring) )
			rename(disambiguated_filename, output_filename);

		else
		{
			sprintf(callbuffer, "%s \"%s\" \"%s\"", decider, disambiguated_filename, decided_filename);
			if ( ! strident(arg_decider, nullstring) )
			{
				strcat(callbuffer, " ");
				strcat(callbuffer, arg_decider);
			}

			if (system(callbuffer))
				printf("Error reported by %s for file %s!\n", decider, input_filename);

			/* run the specified improver, if any */
			/* improver will only run if decider has run, therefore within the if */
			if ( strident(improver, nullstring) )
				rename(decided_filename, output_filename);

			else
			{
				sprintf(callbuffer, "%s \"%s\" \"%s\"", improver, decided_filename, output_filename);
				if ( ! strident(arg_improver, nullstring) )
				{
					strcat(callbuffer, " ");
					strcat(callbuffer, arg_improver);
				}

				if (system(callbuffer))
					printf("Error reported by %s for file %s!\n", improver, input_filename);
			}
		}

		/* delete intermediary files  */
		remove(tokenised_filename);
		remove(analysed_filename);
		remove(disambiguated_filename);
		remove(decided_filename);
		
		printf("Done with file %s !\n", input_filename);
	}




	/* close listsource file */
	if (fclose(listsource) != 0)
	{
		cl_error_file_close(listsource_filename);
		fcloseall();
		return 1;
	}

	/* delete any single-filename listfile */
	if (argc == 3)
		remove(listsource_filename);


	return 0;
}