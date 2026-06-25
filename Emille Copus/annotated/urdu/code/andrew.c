#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <time.h>
#include "andrew.h"

void enter_to_continue(void)
{
	char junk[80];
	fputs("\tPress [ENTER] to continue :  ", stdout);
	gets(junk);
}

void blank_line(int count)
{
	int cntr;
	for (cntr = 0; cntr < count; cntr++)
		puts(" ");
}

void clear_screen(void)
{
	/* MS-DOS / Win32-console specific version */
	system("cls");
	/* Unix specific version 
	system("clear"); */
}

void message(const char *string)
{
	clear_screen();
	puts("\n\n\n\n\n\n\n\n\n\n");
	cenputs(string);
	puts("\n\n\n\n\n\n\n\n\n\n");
	enter_to_continue();
}

int query(const char *string)
{
	clear_screen();
	puts("\n\n\n\n\n\n\n\n\n\n");
	cenputs(string);
	puts("\n\n\n\n\n\n\n\n\n\n");
	return (input_yes_no());
}

void delay(int length)
{
	clock_t goal;

	goal = (length * CLOCKS_PER_SEC) + clock();
	
	while (goal > clock()) ;
}

void error_file_close(const char *file)
{
	clear_screen(), blank_line(11);
	fprintf(stderr, "\n\n                 There was an error closing \"%s\"", file);
	fputs("\n\n                        Abort the program ([Y] / [N]) \?\n\n", stderr);
	blank_line(10);
	if (input_yes_no() == YES)
		exit(1);
}

void error_file_open(const char *file)
{
	clear_screen(), blank_line(10);
	fprintf(stderr, "\n\n                 There was an error opening \"%s\"", file);
	fputs("\n\n                     You may have entered an invalid filename.", stderr);
	fputs("\n\n                        Abort the program ([Y] / [N]) \?\n\n", stderr);
	blank_line(9);
	if (input_yes_no() == YES)
		exit(1);
}

void error_file_read(const char *file)
{
	clear_screen(), blank_line(11);
	fprintf(stderr, "\n\n                There was an error reading from \"%s\"", file);
	fputs("\n\n                        Abort the program ([Y] / [N]) \?\n\n", stderr);
	blank_line(10);
	if (input_yes_no() == YES)
		exit(1);
}

void error_file_write(const char *file)
{
	clear_screen(), blank_line(11);
	fprintf(stderr, "\n\n                 There was an error writing to \"%s\"", file);
	fputs("\n\n                        Abort the program ([Y] / [N]) \?\n\n", stderr);
	blank_line(10);
	if (input_yes_no() == YES)
		exit(1);
}

void error_memory_alloc(void)
{
	clear_screen(), blank_line(11);
	cenfputs("There is a memory allocation error.", stderr);
	fputs("\n\n                        Abort the program ([Y] / [N]) \?\n\n", stderr);
	blank_line(10);
	if (input_yes_no() == YES)
		exit(1);
}


int input_yes_no(void)
{
	char input;
	fputs("\t [[Y]es] or [N]o ?    ", stdout);
	while (1)
	{
		input = getch();
		if (input == 'Y' || input == 'y' || input == '\r')
			return YES;
		if (input == 'N' || input == 'n')
			return NO;
	}
}

int input_integer(void)
{
	int integer;
	fputs("Input an integer and press [ENTER]:  ", stdout);
	scanf("%d", &integer);
	return integer;
}

/* requires an uppercase string argument containing all the acceptable options              */
/* if a double-byte escape keystroke is required, then an escape code must be included      */
/* in the string, in which case the escape character is returned and the next character is  */
/* not taken from the stream. The calling function must deal with the subsequent character. */
/* escape codes: | for 0xE0, ^ for 0x00.                                                    */
/* argument may also be NULL, in which case anythign is returned                            */
char input_one_touch(const char *options)	/* all letters are uppercase	*/
{
	unsigned char input;
	fputs("\tMake a selection :  ", stdout);
	while(1)
	{
		switch (input = getch()) {		
		case 0:		input = '^';	break;
		case 0xE0:	input = '|';	break;
		default:		
			if (input > 0x60 && input < 0x7B)
				input -= 0x20;
		}
		if (options)
		{
			if (!strchr(options, input))
			{
				/* if input is an unwanted control character, skip the next character as well */
				if (input == '^' || input == '|' )
					input = getche();
				continue;
			}
			else
				break;
		}
		else
			break;
	}
	return input;
}

void input_string(char *target)
{
	fputs("Type and then press [ENTER] :  ", stdout);
	gets(target);
}

/* allows maximum input of 256 characters; this CAN overwrite the target string if it's not big enough! */
void input_hidden_string(char *target)
{
	int i;
	fputs("Type and then press [ENTER] :  ", stdout);
	for (i = 0; i < 256; i++)
		if ((*(target+i) = getch()) == '\r')
			break;
	*(target+i) = 0;
}

/* returns 0 if file doesn't exist or user chooses to overwrite, 1 if the file exists	*/
int test_file_write(const char *filename)
{
	FILE *file_test;
	char buffer[100];

	if (file_test = fopen(filename, "rb"))
	{
		fclose(file_test);
		
		clear_screen(), blank_line(10);
		sprintf(buffer, "File %s already exists, do you wish to overwrite?", filename);
		cenputs(buffer);
		cenputs("You will lose all previous data in that file if overwritten.");
		blank_line(11);
		
		fputs("Overwrite?  ", stdout);
		if (input_yes_no() == NO)
			return 1;
	}
	return 0;
}

unsigned char view_file(const char *filename)
{
	/* note: this function only displays text files with 80-char lines.					*/
	/* returns 0 for failure, 1 for success.											*/
	char buffer[200];
	char *control;
	long line_count = 0;
	FILE *source;

	clear_screen();
	if (!(source = fopen(filename, "r")))
	{
		message("The requested ASCII file is not available.");
		return 0;
	}
	while (1)
	{
		if ( (++line_count % 24) == 0 )
		{
			puts(" ");
			enter_to_continue();
			clear_screen();
		}
		if ((control = fgets(buffer, 200, source)) == buffer)
			fputs(buffer, stdout);
		else
			break;
	}
	while (++line_count % 24)
		puts(" ");
	fputs("\n\n\tEnd of file!", stdout);
	enter_to_continue();
	if (fclose(source))
	{
		error_file_close(filename);
		fcloseall();
		return 0;
	}
	return 1;
}

int cenputs(const char *string)
{
	char x, cntr;
	for (cntr = 0, x = (80-strlen(string)) / 2; cntr < x; cntr++)
		if (fputs(" ", stdout) < 0)
			return EOF;
	if (fputs(string, stdout) < 0)
		return EOF;
	if (fputs("\n", stdout) < 0)
		return EOF;
	return 1;
}

int cenfputs(const char *string, FILE *dest)
{
	char x, cntr;
	for (cntr = 0, x = (80-strlen(string)) / 2; cntr < x; cntr++)
		if (fputs(" ", dest) < 0)
			return EOF;
	if (fputs(string, dest) < 0)
		return EOF;
	return 1;
}

/* returns 1 if the strings are identical, 0 if they are not */
int strident(const char *string1, const char *string2)
{
	size_t length;
	size_t x;
	if ((length = strlen(string1)) != strlen(string2))
		return NO;
	for (x = 0; x <= length; x++)
		if (*(string1+x) != *(string2+x))
			return NO;
	return YES;
}
/* returns number of characters that are the same in the strings */
size_t strnident(const char *string1, const char *string2)
{
	size_t cntr = 0;
	while (*(string1) == *(string2++))
	{
		cntr++;
		if (!(*(string1++)))
			break;
	}
	return cntr;
}

/*note: see unicode.c for unicode versions of the above functions */

char *get_output_textname(const char *input)
{
	char *output;

	if (!(output = (char *)malloc(300 * sizeof(char))))
		;
	else
	{
		strcpy(output, input);
		strcat(output, "_output.txt");
	}
	return output;
}

void decaps(char *string)
{
	while (*string)
	{
		if (*string > 0x40 && *string < 0x5b)
			*string += 0x20;
		string++;
	}
}