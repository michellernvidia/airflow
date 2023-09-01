from utils.args import arguments
from utils.fileio import read_config, read_model
from utils.prompts import build_request_prompts, convert_prompt
from utils.tokenizer import get_task_template, get_tokenizer, ids_to_text
from utils.triton import generate_inputs, send_prompt

PROMPT = 'Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.\n\nQuestion: Which NFL team won Super Bowl 50?\n\nAnswer:'

def main():
    args = arguments()
    
    if args.ptuned_model:
        ptuning_mode=True
        model_weights, model_config = read_model(args.ptuned_model)
        config = read_config(model_config)

        request_prompts = build_request_prompts(model_weights, args.taskname)
        request_prompt_lengths, request_prompt_embedding, request_prompt_type = request_prompts

        task_templates, num_virtual_tokens = get_task_template(config, args.taskname)
        tokenizer, pseudo_tokens = get_tokenizer(args.use_nemo, num_virtual_tokens)

        input_ids = convert_prompt(
            args.use_nemo,
            ptuning_mode,
            PROMPT,
            tokenizer,
            task_templates,
            pseudo_tokens,
            args.taskname,
            config
        )

        inputs = generate_inputs(
            args,
            input_ids,
            ptuning_mode,
            request_prompt_embedding,
            request_prompt_lengths,
            request_prompt_type
        )

    else:
        ptuning_mode=False
        num_virtual_tokens=0 
        tokenizer, _ = get_tokenizer(args.use_nemo, num_virtual_tokens)

        #TO DO: have to configure GPTSFTDataset for --use-nemo with SFT + LoRA
        #works with HF tokenizer, but not with NeMo tokenizer. Have to figure out how to send in
        #one prompt at a time and not a list of files to GPTSFTDataset
        #My other question is: do we not have to reformat the prompt into the "{input} {output}" format of the datasets
        #we trained our models (LoRA, SFT) on?
        input_ids = convert_prompt(
            args.use_nemo,
            ptuning_mode,
            PROMPT,
            tokenizer
        )

        inputs = generate_inputs(
            args,
            input_ids,
            ptuning_mode
        )

    response = send_prompt(args.server, args.model_name, inputs)
    text = ids_to_text(tokenizer, response)
    
    print('--------------------RESPONSE TEXT---------------------------')
    if args.ptuned_model:
        print('PROMPT\n\n', text.split('Answer: {}')[0], '\n')
        print('ANSWER\n\n', text.split('Answer: {}')[-1], '\n')
    else:
        print('RESPONSE TEXT\n\n', text)
    print('--------------------END OF RESPONSE TEXT--------------------')


if __name__ == '__main__':
    main()


