const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");
const refreshButton = document.querySelector("#refresh-btn");
const popupForm = document.getElementById("popup-form");
const additionalThoughtsInput = document.getElementById("dislike-additional-thoughts");
const submitButton = document.getElementById("dislike-submit-button");
const closeButton = document.getElementById("dislike-close-button");
const correct_checkbox= document.getElementById("correct_checkbox");
const helpful_checkbox = document.getElementById("helpful_checkbox");
const appropriate_checkbox = document.getElementById("appropriate_checkbox");
popupForm.style.display = "none";

let userText = null;
let discussion_id = null;
let next_message_id = 0;
let conversation = []
let num_responses_since_last_rating = 0;

const loadDataFromLocalstorage = () => {
    // Load saved chats and theme from local storage and apply/add on the page
    const themeColor = localStorage.getItem("themeColor");

    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";

    const defaultText = `<div class="default-text">
                            <h1>A2rchi</h1>
                            <p>Start a conversation and explore the power of A2rchi, specially trained on subMIT.<br> 
                            Your chat history will be displayed here. <br> <br>
                            By using this website, you agree to the <a href="/terms">terms and conditions</a>.</p>
                        </div>`

    chatContainer.innerHTML = localStorage.getItem("all-chats") || defaultText;
    chatContainer.scrollTo(0, chatContainer.scrollHeight); // Scroll to bottom of the chat container
}

const createChatElement = (content, className) => {
    // Create new div and apply chat, specified class and set html content of div
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);
    chatDiv.innerHTML = content;
    return chatDiv; // Return the created chat div
}

const refreshChat = async () => {
    conversation.pop();
    if (next_message_id > 0) {
        next_message_id = next_message_id -1;
    }
    chatContainer.removeChild(chatContainer.lastChild);
    showTypingAnimation();
}

const getChatResponse = async (incomingChatDiv) => {
    const API_URL = "http://t3desk019.mit.edu:7861/api/get_chat_response";
    const pElement = document.createElement("div");

    // Give the p element of the response an id which is equal to the message id
    pElement.setAttribute('id',next_message_id.toString());
    next_message_id = next_message_id + 1

     // Define the properties and data for the API request
     const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            conversation: conversation,
            discussion_id: discussion_id,
        })
    }

     // Send POST request to Flask API, get response and set the response as paragraph element text
     try {
        const response = await (await fetch(API_URL, requestOptions)).json();
        pElement.innerHTML = response.response;
        pElement.classList.add(".default-text");
        conversation.push(["A2rchi", response.response]);
        discussion_id = response.discussion_id ;
    } catch (error) {
        pElement.classList.add("error");
        pElement.textContent = "Oops! Something went wrong while retrieving the response. Please try again.";
    }

    // Remove the typing animation, append the paragraph element and save the chats to local storage
    incomingChatDiv.querySelector(".typing-animation").remove();
    incomingChatDiv.querySelector(".chat-details").appendChild(pElement);
    localStorage.setItem("all-chats", chatContainer.innerHTML);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
}

const copyCode = (copyCodeBtn) => {
    // Copy the text content of the response to the clipboard
    const reponseTextElement = copyCodeBtn.parentElement.parentElement.querySelector(".code-box-body");
    navigator.clipboard.writeText(reponseTextElement.innerText);
}

const copyResponse = (copyBtn) => {
    // Copy the text content of the response to the clipboard
    const reponseTextElement = copyBtn.parentElement.previousElementSibling.querySelector("p");
    navigator.clipboard.writeText(reponseTextElement.textContent);
}

const likeResponse = (likeBtn) => {
    num_responses_since_last_rating = 0;

    const chatContent = likeBtn.parentElement.previousElementSibling.querySelector("p").textContent;

    // fill the image
    const image = likeBtn.querySelector("img");
    image.src = "/static/images/thumbs_up_filled.png"

    // make sure other image is not filled
    const other_image = likeBtn.nextElementSibling.querySelector("img");
    other_image.src = "/static/images/thumbs_down.png";

    const API_URL = "http://t3desk019.mit.edu:7861/api/like";

     // Send an API request with the chat content and discussion ID
     fetch(API_URL, {
        method: "POST", // You may need to adjust the HTTP method
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify({ 
            content: chatContent,
            discussion_id: discussion_id,
            message_id: likeBtn.parentElement.previousElementSibling.querySelector("p").parentElement.id,
        }),
    })
}

const dislikeResponse = (dislikeBtn) => {
    num_responses_since_last_rating = 0;

    const chatContent = dislikeBtn.parentElement.previousElementSibling.querySelector("p").textContent;

    // fill the image
    const image = dislikeBtn.querySelector("img");
    image.src = "/static/images/thumbs_down_filled.png";

    // make sure other image is not filled
    const other_image = dislikeBtn.previousElementSibling.querySelector("img");
    other_image.src = "/static/images/thumbs_up.png";

    const API_URL = "http://t3desk019.mit.edu:7861/api/dislike";

    // Show pop-up form
    popupForm.style.display = "block";

    // Function which handles sending the information in the pop up form to the backend API
    function handleSubmitToAPI() {
        const additionalThoughts = additionalThoughtsInput.value;

        fetch(API_URL, {
            method: "POST", // You may need to adjust the HTTP method
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            body: JSON.stringify({ 
                content: chatContent,
                discussion_id: discussion_id,
                message_id: dislikeBtn.parentElement.previousElementSibling.querySelector("p").parentElement.id,
                message: additionalThoughts,
                incorrect: correct_checkbox.checked,
                unhelpful: helpful_checkbox.checked,
                inappropriate: appropriate_checkbox.checked,
            }),
        });

        //hide pop up formi
        popupForm.style.display = "none";
    }

    //wait for user to submit response or close the additional feedback window
    submitButton.addEventListener("click", handleSubmitToAPI);
    closeButton.addEventListener("click", handleSubmitToAPI);
}

const closeFeedback = (closeBtn) => {
    //hide pop up formi
    popupForm.style.display = "none";
}

const showTypingAnimation = () => {
    // Display the typing animation and call the getChatResponse function
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/a2rchi.png" alt="chatbot-img">
                        <div class="typing-animation">
                            <div class="typing-dot" style="--delay: 0.2s"></div>
                            <div class="typing-dot" style="--delay: 0.3s"></div>
                            <div class="typing-dot" style="--delay: 0.4s"></div>
                        </div>
                    </div>
                    <div class="button-container">
                        <button onclick="likeResponse(this)" class="material-button">
                            <img src="/static/images/thumbs_up.png" alt="Like" width="30" height="30">
                        </button>
                        <button onclick="dislikeResponse(this)" class="material-button">
                            <img src="/static/images/thumbs_down.png" alt="Dislike" width="30" height="30">
                        </button>
                    <div>
                </div>`;
    // Create an incoming chat div with typing animation and append it to chat container
    const incomingChatDiv = createChatElement(html, "incoming");
    chatContainer.appendChild(incomingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    getChatResponse(incomingChatDiv);
    showFeedbackRequest();
}

const showFeedbackRequest = () => {
    // Display a message from A2rchi to ask the user to give feedback

    num_responses_since_last_rating = num_responses_since_last_rating + 1;

    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/a2rchi.png" alt="chatbot-img">
                        <div class=".default-text">
                            <p>I've noticed you haven't rated any of my responses in awhile. Rating responses is crucial because it not only helps me improve, but it also ensures that this project remains open source and freely accessible for everyone. Your input is highly valuable in supporting the A2rchi mission! </p>
                        </div>
                    </div>
                </div>`;
    
    // Create an incoming chat div with feedback request and append it to chat container
    if (num_responses_since_last_rating > 2) {
        const incomingChatDiv = createChatElement(html, "incoming");
        chatContainer.appendChild(incomingChatDiv);
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
        num_responses_since_last_rating = 0;
    }
}

const handleOutgoingChat = () => {
    userText = chatInput.value.trim(); // Get chatInput value and remove extra spaces
    if(!userText) return; // If chatInput is empty return from here
    conversation.push(["User", userText])

    // Clear the input field and reset its height
    chatInput.value = "";
    chatInput.style.height = `${initialInputHeight}px`;

    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/user.svg" alt="user-img">
                        <p>${userText}</p>
                    </div>
                </div>`;

    // Create an outgoing chat div with user's message and append it to chat container
    const outgoingChatDiv = createChatElement(html, "outgoing");
    chatContainer.querySelector(".default-text")?.remove();
    chatContainer.appendChild(outgoingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    setTimeout(showTypingAnimation, 500);
}

deleteButton.addEventListener("click", () => {
    // Remove the chats from local storage and call loadDataFromLocalstorage function
    if(confirm("Are you sure you want to delete all the chats?")) {
        conversation = []
        discussion_id = null
        next_message_id = 0;
        localStorage.removeItem("all-chats");
        loadDataFromLocalstorage();
    }
});

refreshButton.addEventListener("click", () => {
    refreshChat();
});

themeButton.addEventListener("click", () => {
    // Toggle body's class for the theme mode and save the updated theme to the local storage 
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});

const initialInputHeight = chatInput.scrollHeight;

chatInput.addEventListener("input", () => {   
    // Adjust the height of the input field dynamically based on its content
    chatInput.style.height =  `${initialInputHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If the Enter key is pressed without Shift and the window width is larger 
    // than 800 pixels, handle the outgoing chat
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleOutgoingChat();
    }
});

loadDataFromLocalstorage();
sendButton.addEventListener("click", handleOutgoingChat);
