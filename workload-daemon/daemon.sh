#!/bin/bash

mkdir -p workloads/1.to_run workloads/2.queued workloads/3.running workloads/4.executed workloads/4.failed

if [[ -z $MAX_RAM ]]; then
    MAX_RAM="0.5"
fi
if [[ -z $MAX_VRAM ]]; then
    MAX_VRAM="0.5"
fi
if [[ -z $RAMP_UP_TIME ]]; then
    RAMP_UP_TIME="30" # seconds between runs
fi

echo "Default max RAM is $MAX_RAM and max VRAM is $MAX_VRAM"

TOTAL_RAM=$(free | awk '/Mem/{printf("%.2f"), $2/1024/1024}' | sed 's/,/\./')
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
TOTAL_VRAM=$(echo "scale=2; $TOTAL_VRAM / 1024" | bc)
MIN_AVAILABLE_RAM=$(echo "scale=2; $TOTAL_RAM * $MAX_RAM" | bc)
MIN_AVAILABLE_VRAM=$(echo "scale=2; $TOTAL_VRAM * $MAX_VRAM" | bc)

calculate_ram_availability() {
    free | awk '/Mem/{printf("%.2f"), $2/1024/1024}' | sed 's/,/\./'
}

calculate_vram_availability() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1
}

no_scripts_printed=false

while true; do
  # Check if there are any scripts in the "workloads/to_run" folder
  if [ -n "$(ls -A workloads/1.to_run)" ]; then
    # Get the first script in the "workloads/1.to_run" folder
    script=$(ls workloads/1.to_run | head -n 1)

    echo "---- Moving $script to 2.queued"
    mv "workloads/1.to_run/$script" "workloads/2.queued/$script"
    echo "---- Queued $script"

    min_available_ram=$MIN_AVAILABLE_RAM
    min_available_vram=$MIN_AVAILABLE_VRAM
    ramp_up_time=$RAMP_UP_TIME
    # Read the second line of the script to check for hardware requirements
    req_line=$(sed -n '2p' "workloads/2.queued/$script")
    # Check if the second line contains hardware requirements
    if [[ $req_line == '# {"requirements": {'* ]]; then
        # Extract RAM and VRAM requirements from the line using jq
        req_ram=$(echo $req_line | sed 's/^# //' | jq -r '.requirements.ram // empty')
        req_vram=$(echo $req_line | sed 's/^# //' | jq -r '.requirements.vram // empty')
        req_ramp_up_time=$(echo $req_line | sed 's/^# //' | jq -r '.requirements.ramp_up_time // empty')

        # Update min_available_ram and min_available_vram if requirements are found
        if [[ -n $req_ram ]]; then
            min_available_ram=$req_ram
        fi
        if [[ -n $req_vram ]]; then
            min_available_vram=$req_vram
        fi
        if [[ -n $req_ramp_up_time ]]; then
            ramp_up_time=$req_ramp_up_time
        fi
    fi

    echo "---- Script $script requires min RAM: $min_available_ram and min VRAM: $min_available_vram"

    ram_availability=$(calculate_ram_availability)
    vram_availability=$(calculate_vram_availability)

    echo "---- Available RAM: $ram_availability GB / Available VRAM: $vram_availability GB"

    while true; do
      if (( $(echo "$ram_availability > $min_available_ram" | bc -l) )) && (( $(echo "$vram_availability > $min_available_vram" | bc -l) )); then
        # If there is enough RAM and VRAM available, break the loop and continue with the execution
        echo "---- Running $script"
        break
      else
        # If there is not enough RAM or VRAM available, print current timestamp and sleep
        echo "---- Waiting. Available RAM: $ram_availability GB / Available VRAM: $vram_availability GB / Required RAM: $min_available_ram GB / Required VRAM: $min_available_vram GB"
        date
        sleep 180

        ram_availability=$(calculate_ram_availability)
        vram_availability=$(calculate_vram_availability)
      fi
    done

    echo "---- Moving $script to 3.running"
    mv "workloads/2.queued/$script" "workloads/3.running/$script"

    # Execute the script in the background and move it to the appropriate folder based on the exit status
    (
        setsid bash -c "
            source workloads/3.running/$script
            exit_status=\$?
            if [ \$exit_status -eq 0 ]; then
                echo \"---- Moving $script to 4.executed\"
                mv workloads/3.running/$script workloads/4.executed/$script
            else
                echo \"---- Moving $script to 4.failed\"
                mv workloads/3.running/$script workloads/4.failed/$script
            fi
        " &>/dev/null &
    )
    echo "---- Waiting $ramp_up_time s for the script to ramp up"
  
    # trap 'echo "Interrupt received, waiting for script to finish..."; wait $script_pid; exit 0' INT
    sleep $ramp_up_time # time to wait for the training begin using resources
    no_scripts_printed=false
  else
    # If there are no scripts in the "workloads/1.to_run" folder, print current timestamp and sleep
    if ! $no_scripts_printed; then
      echo "---- No more scripts to run"
      date
      no_scripts_printed=true
    fi
    sleep 10
  fi
done